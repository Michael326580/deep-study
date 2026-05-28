function build_obc_11kw_afe_dab_v2_simscape()
% ============================================================
% V2: Auto-build Simulink model with Simscape Electrical AFE plant
% 11kW OBC skeleton: 3-phase AFE (switching plant) + DAB controller placeholder
% ------------------------------------------------------------
% Generated model:
%   OBC_11kW_AFE_DAB_V2.slx
% ============================================================

clc; close all;

%% Parameters
P = init_obc_params_v2();
assignin('base','P',P);

mdl = 'OBC_11kW_AFE_DAB_V2';
if bdIsLoaded(mdl), close_system(mdl,0); end
new_system(mdl); open_system(mdl);

set_param(mdl,'Solver','ode23t','StopTime','0.15');

% Top-level blocks
add_block('simulink/Ports & Subsystems/Subsystem',[mdl '/Control_AFE_dq_SPWM'],...
    'Position',[80 60 380 300]);
add_block('simulink/Ports & Subsystems/Subsystem',[mdl '/Control_DAB_SPS'],...
    'Position',[80 340 380 500]);
add_block('simulink/Ports & Subsystems/Subsystem',[mdl '/Plant_Simscape_AFE'],...
    'Position',[430 60 900 500]);
add_block('simulink/Sinks/Scope',[mdl '/Scope'],'Position',[950 90 1100 260]);
add_block('simulink/Sources/Constant',[mdl '/Vdc_ref'],'Value','P.Vdc_ref',...
    'Position',[20 120 60 145]);
add_block('simulink/Sources/Constant',[mdl '/Vbat_ref'],'Value','P.Vbat_nom',...
    'Position',[20 390 60 415]);

% Build subsystems
build_control_afe([mdl '/Control_AFE_dq_SPWM']);
build_control_dab([mdl '/Control_DAB_SPS']);
build_simscape_afe_plant([mdl '/Plant_Simscape_AFE']);

% Wire top level
add_line(mdl,'Vdc_ref/1','Control_AFE_dq_SPWM/4','autorouting','on');
add_line(mdl,'Plant_Simscape_AFE/1','Control_AFE_dq_SPWM/1','autorouting','on'); % vabc
add_line(mdl,'Plant_Simscape_AFE/2','Control_AFE_dq_SPWM/2','autorouting','on'); % iabc
add_line(mdl,'Plant_Simscape_AFE/3','Control_AFE_dq_SPWM/3','autorouting','on'); % Vdc
add_line(mdl,'Control_AFE_dq_SPWM/5','Plant_Simscape_AFE/1','autorouting','on'); % gates

add_line(mdl,'Vbat_ref/1','Control_DAB_SPS/1','autorouting','on');
add_line(mdl,'Plant_Simscape_AFE/4','Control_DAB_SPS/2','autorouting','on');

add_line(mdl,'Control_AFE_dq_SPWM/6','Scope/1','autorouting','on'); % id_ref
add_line(mdl,'Control_AFE_dq_SPWM/7','Scope/2','autorouting','on'); % id
add_line(mdl,'Control_DAB_SPS/3','Scope/3','autorouting','on');      % phi
add_line(mdl,'Plant_Simscape_AFE/3','Scope/4','autorouting','on');   % Vdc

save_system(mdl,[mdl '.slx']);
open_system(mdl);
fprintf('Model generated: %s.slx\n',mdl);

end

function P = init_obc_params_v2()
P.P_rated = 11e3;
P.Vll_rms = 220;
P.f_grid = 50;
P.w_grid = 2*pi*P.f_grid;
P.Vdc_ref = 700;
P.Vbat_nom = 380;
P.Vbat_min = 250;
P.Vbat_max = 450;
P.fsw_afe = 20e3;
P.Ts = 1/P.fsw_afe;
P.Lf = 2.2e-3;
P.Rf = 0.06;
P.Cdc = 2200e-6;
P.Kp_v = 0.5; P.Ki_v = 20;
bi = 2*pi*(P.fsw_afe/20);
P.Kp_i = P.Lf*bi; P.Ki_i = P.Rf*bi;
P.id_ref_max = 120; P.id_ref_min = -120;
P.Kp_dab = 0.02; P.Ki_dab = 5;
P.phi_max = 0.45*pi; P.phi_min = -0.45*pi;
P.Vph_peak = sqrt(2)*(P.Vll_rms/sqrt(3));
end

function build_control_afe(sys)
open_system(sys);
% Ports
add_block('simulink/Ports & Subsystems/In1',[sys '/vabc'],'Position',[20 40 50 60]);
add_block('simulink/Ports & Subsystems/In1',[sys '/iabc'],'Position',[20 80 50 100]);
add_block('simulink/Ports & Subsystems/In1',[sys '/Vdc'],'Position',[20 120 50 140]);
add_block('simulink/Ports & Subsystems/In1',[sys '/Vdc_ref'],'Position',[20 160 50 180]);
add_block('simulink/Ports & Subsystems/Out1',[sys '/gateABC'],'Position',[860 80 890 100]);
add_block('simulink/Ports & Subsystems/Out1',[sys '/id_ref_o'],'Position',[860 150 890 170]);
add_block('simulink/Ports & Subsystems/Out1',[sys '/id_o'],'Position',[860 200 890 220]);

add_block('simulink/Sources/Clock',[sys '/clk'],'Position',[80 250 110 270]);
add_block('simulink/Math Operations/Gain',[sys '/wg'],'Gain','P.w_grid','Position',[125 245 180 275]);
add_line(sys,'clk/1','wg/1');

add_block('simulink/User-Defined Functions/MATLAB Function',[sys '/abc2dq'],'Position',[220 40 340 140]);
set_param([sys '/abc2dq'],'Script',sprintf(['function [id,iq,vd,vq]=fcn(iabc,vabc,th)\n' ...
'ia=iabc(1);ib=iabc(2);ic=iabc(3);va=vabc(1);vb=vabc(2);vc=vabc(3);\n' ...
'c=cos(th);s=sin(th);c2=cos(th-2*pi/3);s2=sin(th-2*pi/3);c3=cos(th+2*pi/3);s3=sin(th+2*pi/3);\n' ...
'id=2/3*(ia*c+ib*c2+ic*c3); iq=-2/3*(ia*s+ib*s2+ic*s3);\n' ...
'vd=2/3*(va*c+vb*c2+vc*c3); vq=-2/3*(va*s+vb*s2+vc*s3);\n']));
add_line(sys,'iabc/1','abc2dq/1'); add_line(sys,'vabc/1','abc2dq/2'); add_line(sys,'wg/1','abc2dq/3');
add_block('simulink/Signal Routing/Demux',[sys '/dm'],'Outputs','4','Position',[370 45 375 145]);
add_line(sys,'abc2dq/1','dm/1');

add_block('simulink/Math Operations/Sum',[sys '/sumv'],'Inputs','+-','Position',[220 165 240 195]);
add_block('simulink/Continuous/PID Controller',[sys '/PIv'],'P','P.Kp_v','I','P.Ki_v','D','0','Position',[260 158 330 202]);
add_block('simulink/Discontinuities/Saturation',[sys '/satid'],'UpperLimit','P.id_ref_max','LowerLimit','P.id_ref_min','Position',[355 165 430 195]);
add_line(sys,'Vdc_ref/1','sumv/1'); add_line(sys,'Vdc/1','sumv/2'); add_line(sys,'sumv/1','PIv/1'); add_line(sys,'PIv/1','satid/1');

add_block('simulink/Sources/Constant',[sys '/iq_ref'],'Value','0','Position',[440 220 470 245]);
add_block('simulink/Math Operations/Sum',[sys '/sumid'],'Inputs','+-','Position',[490 50 510 80]);
add_block('simulink/Math Operations/Sum',[sys '/sumiq'],'Inputs','+-','Position',[490 110 510 140]);
add_block('simulink/Continuous/PID Controller',[sys '/PIid'],'P','P.Kp_i','I','P.Ki_i','D','0','Position',[530 42 610 88]);
add_block('simulink/Continuous/PID Controller',[sys '/PIiq'],'P','P.Kp_i','I','P.Ki_i','D','0','Position',[530 102 610 148]);
add_line(sys,'satid/1','sumid/1'); add_line(sys,'iq_ref/1','sumiq/1'); add_line(sys,'dm/1','sumid/2'); add_line(sys,'dm/2','sumiq/2');
add_line(sys,'sumid/1','PIid/1'); add_line(sys,'sumiq/1','PIiq/1');

add_block('simulink/Math Operations/Sum',[sys '/sumvd'],'Inputs','++','Position',[635 50 655 80]);
add_block('simulink/Math Operations/Sum',[sys '/sumvq'],'Inputs','++','Position',[635 110 655 140]);
add_line(sys,'PIid/1','sumvd/1'); add_line(sys,'dm/3','sumvd/2');
add_line(sys,'PIiq/1','sumvq/1'); add_line(sys,'dm/4','sumvq/2');

add_block('simulink/User-Defined Functions/MATLAB Function',[sys '/dq2abc'],'Position',[680 45 780 145]);
set_param([sys '/dq2abc'],'Script',sprintf(['function [va,vb,vc]=fcn(vd,vq,th)\n' ...
'va=vd*cos(th)-vq*sin(th);\n' ...
'vb=vd*cos(th-2*pi/3)-vq*sin(th-2*pi/3);\n' ...
'vc=vd*cos(th+2*pi/3)-vq*sin(th+2*pi/3);\n']));
add_line(sys,'sumvd/1','dq2abc/1'); add_line(sys,'sumvq/1','dq2abc/2'); add_line(sys,'wg/1','dq2abc/3');

add_block('simulink/Sources/Repeating Sequence',[sys '/car'],'rep_seq_t','[0 1/P.fsw_afe]','rep_seq_y','[-1 1]','Position',[680 180 760 205]);
for k=1:3
    c=char('a'+k-1);
    add_block('simulink/Math Operations/Gain',[sys '/n_' c],'Gain','2/P.Vdc_ref','Position',[790 25+50*(k-1) 840 50+50*(k-1)]);
    add_block('simulink/Logic and Bit Operations/Relational Operator',[sys '/cmp_' c],'Operator','>=','Position',[850 25+50*(k-1) 900 50+50*(k-1)]);
end
add_line(sys,'dq2abc/1','n_a/1'); add_line(sys,'dq2abc/2','n_b/1'); add_line(sys,'dq2abc/3','n_c/1');
add_line(sys,'n_a/1','cmp_a/1'); add_line(sys,'n_b/1','cmp_b/1'); add_line(sys,'n_c/1','cmp_c/1');
add_line(sys,'car/1','cmp_a/2'); add_line(sys,'car/1','cmp_b/2'); add_line(sys,'car/1','cmp_c/2');
add_block('simulink/Signal Routing/Mux',[sys '/muxg'],'Inputs','3','Position',[920 45 925 145]);
add_line(sys,'cmp_a/1','muxg/1'); add_line(sys,'cmp_b/1','muxg/2'); add_line(sys,'cmp_c/1','muxg/3');
add_line(sys,'muxg/1','gateABC/1');
add_line(sys,'satid/1','id_ref_o/1'); add_line(sys,'dm/1','id_o/1');
end

function build_control_dab(sys)
open_system(sys);
add_block('simulink/Ports & Subsystems/In1',[sys '/Vbat_ref'],'Position',[20 40 50 60]);
add_block('simulink/Ports & Subsystems/In1',[sys '/Vbat'],'Position',[20 90 50 110]);
add_block('simulink/Ports & Subsystems/Out1',[sys '/phi'],'Position',[300 65 330 85]);
add_block('simulink/Math Operations/Sum',[sys '/sum'],'Inputs','+-','Position',[80 58 100 88]);
add_block('simulink/Continuous/PID Controller',[sys '/PI'],'P','P.Kp_dab','I','P.Ki_dab','D','0','Position',[120 52 190 92]);
add_block('simulink/Discontinuities/Saturation',[sys '/sat'],'UpperLimit','P.phi_max','LowerLimit','P.phi_min','Position',[220 58 280 88]);
add_line(sys,'Vbat_ref/1','sum/1'); add_line(sys,'Vbat/1','sum/2'); add_line(sys,'sum/1','PI/1'); add_line(sys,'PI/1','sat/1'); add_line(sys,'sat/1','phi/1');
end

function build_simscape_afe_plant(sys)
open_system(sys);
% IO ports
add_block('simulink/Ports & Subsystems/In1',[sys '/gateABC'],'Position',[20 260 50 280]);
add_block('simulink/Ports & Subsystems/Out1',[sys '/vabc_meas'],'Position',[900 50 930 70]);
add_block('simulink/Ports & Subsystems/Out1',[sys '/iabc_meas'],'Position',[900 100 930 120]);
add_block('simulink/Ports & Subsystems/Out1',[sys '/Vdc_meas'],'Position',[900 150 930 170]);
add_block('simulink/Ports & Subsystems/Out1',[sys '/Vbat_meas'],'Position',[900 200 930 220]);

% Use Specialized Power Systems blocks for robust availability
add_block('powerlib/Elements/Three-Phase Source',[sys '/Grid3ph'],'Position',[80 40 170 110]);
set_param([sys '/Grid3ph'],'Vrms','P.Vll_rms','Frequency','P.f_grid');

add_block('powerlib/Power Electronics/Universal Bridge',[sys '/AFE_Bridge'],'Position',[330 70 430 180]);
set_param([sys '/AFE_Bridge'],'PowerElectronicDevice','IGBT/Diodes','NumberOfBridgeArms','3','SnubberResistance','1e5','SnubberCapacitance','1e-9');

add_block('powerlib/Elements/Three-Phase Series RLC Branch',[sys '/Lf_Rf'],'Position',[230 50 300 120]);
set_param([sys '/Lf_Rf'],'BranchType','RL','Resistance','P.Rf','Inductance','P.Lf','Capacitance','0');

add_block('powerlib/Elements/Capacitor',[sys '/Cdc'],'Position',[520 80 560 130]);
set_param([sys '/Cdc'],'Capacitance','P.Cdc','InitialVoltage','P.Vdc_ref');

add_block('powerlib/Elements/Series RLC Branch',[sys '/Rload_dc'],'Position',[600 80 680 130]);
set_param([sys '/Rload_dc'],'BranchType','R','Resistance','(P.Vdc_ref^2)/P.P_rated');

add_block('powerlib/Measurements/Three-Phase V-I Measurement',[sys '/VI_meas'],'Position',[180 40 220 120]);
add_block('powerlib/Measurements/Voltage Measurement',[sys '/Vdc_sensor'],'Position',[570 150 620 180]);

add_block('powerlib/Elements/Ground',[sys '/gnd'],'Position',[710 170 730 190]);
add_block('powerlib/Powergui',[sys '/powergui'],'Position',[40 330 120 370]);

% Gate adaptation: 3 gates -> 6 gates with complement
add_block('simulink/Signal Routing/Demux',[sys '/dm_gate'],'Outputs','3','Position',[90 245 95 315]);
add_block('simulink/Logic and Bit Operations/Logical Operator',[sys '/notA'],'Operator','NOT','Position',[140 245 170 265]);
add_block('simulink/Logic and Bit Operations/Logical Operator',[sys '/notB'],'Operator','NOT','Position',[140 275 170 295]);
add_block('simulink/Logic and Bit Operations/Logical Operator',[sys '/notC'],'Operator','NOT','Position',[140 305 170 325]);
add_block('simulink/Signal Routing/Mux',[sys '/mux6'],'Inputs','6','Position',[220 245 225 335]);

add_line(sys,'gateABC/1','dm_gate/1');
add_line(sys,'dm_gate/1','notA/1'); add_line(sys,'dm_gate/2','notB/1'); add_line(sys,'dm_gate/3','notC/1');
add_line(sys,'dm_gate/1','mux6/1'); add_line(sys,'notA/1','mux6/2');
add_line(sys,'dm_gate/2','mux6/3'); add_line(sys,'notB/1','mux6/4');
add_line(sys,'dm_gate/3','mux6/5'); add_line(sys,'notC/1','mux6/6');

% Electrical connections (SPS)
add_line(sys,'Grid3ph/1','VI_meas/1');
add_line(sys,'VI_meas/1','Lf_Rf/1');
add_line(sys,'Lf_Rf/1','AFE_Bridge/1');
add_line(sys,'mux6/1','AFE_Bridge/2');

add_line(sys,'AFE_Bridge/3','Cdc/1');
add_line(sys,'AFE_Bridge/4','Cdc/2');
add_line(sys,'Cdc/1','Rload_dc/1');
add_line(sys,'Cdc/2','Rload_dc/2');
add_line(sys,'Cdc/2','gnd/1');

add_line(sys,'Cdc/1','Vdc_sensor/1');
add_line(sys,'Cdc/2','Vdc_sensor/2');

% Measurement routing
% VI_meas outputs: [Vabc] and [Iabc] via Simulink ports depending version
add_block('simulink/Signal Routing/Demux',[sys '/dm_vi'],'Outputs','2','Position',[760 60 765 120]);
add_line(sys,'VI_meas/2','dm_vi/1');
add_line(sys,'dm_vi/1','vabc_meas/1');
add_line(sys,'dm_vi/2','iabc_meas/1');
add_line(sys,'Vdc_sensor/1','Vdc_meas/1');

% Vbat placeholder output
add_block('simulink/Sources/Constant',[sys '/Vbat_const'],'Value','P.Vbat_nom','Position',[780 200 850 225]);
add_line(sys,'Vbat_const/1','Vbat_meas/1');
end
