# SC25-STZ
Artifacts of SC'25 submission "STZ: A High Quality and High Speed Streaming Lossy Compression Framework for Scientific Data"

## Minimum system & software libraries requirements
OS: Linux (Ubuntu 20.04 is recommended)

Memory: >= 192 GB RAM

Processor: >= 16 cores

Storage: >= 128 GBs

gcc/9.4.0 (or 9.3.0)

cmake (>= 3.23)

python/3.8

### Step 1: Install Singularity
Install [Singularity](https://singularity-tutorial.github.io/01-installation/)

### Step 2: Download, build, and run the pre-built Singularity image file via GitHub (need root privilege)
```
git clone https://github.com/FabioGrosso/SC25-STZ-image
cat SC25-STZ-image/stz.part.* > stz.sif
sudo singularity build --sandbox stz stz.sif
sudo singularity shell --writable stz
```


### Step 3.1: Compress and decompress the **Nyx** dataset using our STZ, SPERR, SZ3, ZFP, and MGARD, and evaluate the data quality and compression ratio under both **high** and **low** CRs. (1 minute).
```
$ bash psnr-nyx.sh
$ bash psnr-nyx-2.sh
```

### Step 3.2: Compress and decompress the **WarpX** dataset using our STZ, SPERR, SZ3, ZFP, and MGARD, and evaluate the data quality and compression ratio under both **high** and **low** CRs. (1 minute).
```
$ bash psnr-wpx.sh
$ bash psnr-wpx-2.sh
```
### Step 3.3: Compress and decompress the **Magnetic** **Reconnection** dataset using our STZ, SPERR, SZ3, ZFP, and MGARD, and evaluate the data quality and compression ratio under both **high** and **low** CRs. (1 minute).
```
$ bash psnr-mag.sh
$ bash psnr-mag-2.sh
```
### Step 3.4: Compress and decompress the **Miranda** dataset using our STZ, SPERR, SZ3, ZFP, and MGARD, and evaluate the data quality and compression ratio under both **high** and **low** CRs. (1 minute).
```
$ bash psnr-mrd.sh
$ bash psnr-mrd-2.sh
```


### Baseline compressors: [SZ3](https://github.com/szcompressor/SZ3.git); [ZFP](https://github.com/LLNL/zfp); [SPERR](https://github.com/NCAR/SPERR); [MGARDX](https://github.com/CODARcode/MGARD/tree/master).

