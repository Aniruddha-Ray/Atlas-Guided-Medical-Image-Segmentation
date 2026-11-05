checkpoint experiments/exp_04_pipeline/chunk_150_50_e100_swin_prior/checkpoint_best.pth
50 validation cases, 96^3 grid

| variant | mean Dice | change vs base |
|---|---|---|
| base | 0.6963 | +0.0000 |
| cc | 0.6972 | +0.0010 |
| hu0.25 | 0.6962 | -0.0001 |
| hu0.5 | 0.6931 | -0.0031 |
| hu1 | 0.6809 | -0.0154 |
| hu2 | 0.6635 | -0.0328 |
| hu0.25+cc | 0.6973 | +0.0011 |
| hu0.5+cc | 0.6943 | -0.0020 |
| hu1+cc | 0.6817 | -0.0145 |
| hu2+cc | 0.6645 | -0.0317 |
| diff3 | 0.6957 | -0.0005 |
| diff8 | 0.6864 | -0.0098 |
| diff3+cc | 0.6956 | -0.0006 |
| diff8+cc | 0.6852 | -0.0110 |
| hu0.25+diff3+cc | 0.6961 | -0.0002 |
| hu0.25+diff8+cc | 0.6844 | -0.0118 |
| hu0.5+diff3+cc | 0.6930 | -0.0032 |
| hu0.5+diff8+cc | 0.6806 | -0.0157 |
| hu1+diff3+cc | 0.6807 | -0.0156 |
| hu1+diff8+cc | 0.6681 | -0.0281 |
| hu2+diff3+cc | 0.6631 | -0.0331 |
| hu2+diff8+cc | 0.6508 | -0.0455 |

per organ: base against the best variant (hu0.25+cc)

| organ | base | best | change |
|---|---|---|---|
| spleen | 0.849 | 0.857 | +0.007 |
| r_kidney | 0.862 | 0.861 | -0.001 |
| l_kidney | 0.823 | 0.820 | -0.003 |
| gallbladder | 0.577 | 0.573 | -0.004 |
| esophagus | 0.577 | 0.578 | +0.001 |
| liver | 0.917 | 0.916 | -0.001 |
| stomach | 0.693 | 0.701 | +0.008 |
| aorta | 0.802 | 0.803 | +0.001 |
| ivc | 0.729 | 0.727 | -0.002 |
| pancreas | 0.601 | 0.604 | +0.003 |
| r_adrenal | 0.494 | 0.495 | +0.001 |
| l_adrenal | 0.432 | 0.433 | +0.001 |
