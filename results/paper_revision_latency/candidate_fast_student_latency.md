# Candidate fast student latency screening

- input: `[1, 2, 2048]`
- These are untrained models. Use this only to select architectures for retraining.

| Candidate | Params | MACs(M) | Mean ms | Median ms | P95 ms |
|---|---:|---:|---:|---:|---:|
| tiny_dw_w8_d3 | 1241 | 0.231 | 0.9905 | 0.8430 | 1.7636 |
| tiny_dw_w8_d4 | 1673 | 0.278 | 1.5711 | 1.4977 | 2.4740 |
| tiny_dw_w16_d4 | 4305 | 0.753 | 1.8465 | 1.7024 | 3.0563 |
| plain_cnn_w8 | 6881 | 1.655 | 0.6618 | 0.6324 | 1.0721 |
| plain_cnn_w16 | 26817 | 6.324 | 0.9216 | 0.8790 | 1.4919 |