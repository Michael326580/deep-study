# Fair latency benchmark report

## Environment

- python: `3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]`
- platform: `Windows-10-10.0.19045-SP0`
- processor: `Intel64 Family 6 Model 165 Stepping 2, GenuineIntel`
- torch: `2.10.0+cpu`
- device: `cpu`
- cuda_available: `False`
- cuda_device_name: `NA`
- torch_num_threads: `1`
- torch_num_interop_threads: `1`
- precision: `FP32`

## Student model

- Checkpoint: `D:\桌面\deep study\results\models\runs\attention_res1d_model\best_attention_res1d.pth`
- Model config: `{'name': 'DeepAttnRes1DRegressor', 'base_width': 32, 'dropout': 0.15}`
- Parameters: `1304001`
- Estimated Conv/Linear MACs for batch=1: `111684736`
- Input source: `real sample from D:\桌面\deep study\data\processed\dataset_distill.npz`

### pytorch_eager

- batch=1, input=[1, 2, 2048], mean=10.7701 ms/batch, median=9.7025 ms/batch, mean/sample=10.7701 ms, throughput=92.85 samples/s
- batch=8, input=[8, 2, 2048], mean=41.1635 ms/batch, median=39.1806 ms/batch, mean/sample=5.1454 ms, throughput=194.35 samples/s
- batch=32, input=[32, 2, 2048], mean=158.3296 ms/batch, median=156.0894 ms/batch, mean/sample=4.9478 ms, throughput=202.11 samples/s

### torchscript_frozen

- batch=1, input=[1, 2, 2048], mean=7.0034 ms/batch, median=6.9707 ms/batch, mean/sample=7.0034 ms, throughput=142.79 samples/s
- batch=8, input=[8, 2, 2048], mean=33.7102 ms/batch, median=31.6716 ms/batch, mean/sample=4.2138 ms, throughput=237.32 samples/s
- batch=32, input=[32, 2, 2048], mean=148.2165 ms/batch, median=143.4006 ms/batch, mean/sample=4.6318 ms, throughput=215.90 samples/s

## Teacher/PCCC timing

- Segment-level core: `{"mean_ms": 0.9832155, "std_ms": 0.2989642751211986, "median_ms": 0.90835, "p05_ms": 0.705195, "p95_ms": 1.6845149999999989, "min_ms": 0.666, "max_ms": 2.0851, "available": true, "source": "D:\\桌面\\deep study\\data\\processed\\dataset_distill.npz", "repeat": 200, "warmup": 20}`
- File-level full teacher: `{"mean_ms": 18.23617, "std_ms": 2.5563223839397837, "median_ms": 17.7163, "p05_ms": 14.464550000000001, "p95_ms": 22.869125, "min_ms": 13.81, "max_ms": 23.1337, "available": true, "n_files": 20, "raw_dir": "D:\\桌面\\deep study\\data\\raw\\1.22"}`

## Interpretation rule

- Cite only comparisons obtained under the same hardware, same input length, same precision, and clearly specified batch size.
- If the student is slower on CPU, the paper should not claim speed superiority. It can still claim a deployment-oriented surrogate model only after optimized inference is demonstrated.