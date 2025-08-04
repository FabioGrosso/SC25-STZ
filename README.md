# SC25-STZ
Artifacts of SC'25 submission "STZ: A High Quality and High Speed Streaming Lossy Compression Framework for Scientific Data". While preparing the artifacts, we executed them on a single node from the Chameleon Cloud, equipped with two Intel Xeon Gold 6242 CPUs and 192 GB of memory (specifically, compute_icelake_r650 configuration).

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

### Step 2: Download and run the pre-built Singularity image file via GitHub (need root privilege), and navigate to the target directory.
```
git clone https://github.com/FabioGrosso/SC25-STZ-image
cat SC25-STZ-image/stz.part.* > stz.sif
sudo singularity build --sandbox stz stz.sif
sudo singularity shell --writable stz
cd /home/src
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
### Step 3.4: Compress and decompress the **Miranda** dataset using our STZ, SPERR, SZ3, ZFP, and MGARD, and evaluate the data quality and compression ratio under both **high** and **low** CRs. (6 minutes).
```
$ bash psnr-mrd.sh
$ bash psnr-mrd-2.sh
```

### Step 4.1: Compress and decompress the Magnetic Reconnection dataset using our STZ, SPERR, SZ3, ZFP, and MGARD in serial and OpenMP modes, and evaluate the speed (3 minutes).
```
bash time-mag-se.sh
bash time-mag-omp.sh
```
### Step 4.2: Compress and decompress the WarpX dataset using our STZ, SPERR, SZ3, ZFP, and MGARD in serial and OpenMP modes, and evaluate the speed (3 minutes).
```
bash time-wpx-se.sh
bash time-wpx-omp.sh
```
### Step 4.3: Compress and decompress the Nyx dataset using our STZ, SPERR, SZ3, ZFP, and MGARD in serial and OpenMP modes, and evaluate the speed (3 minutes).
```
bash time-nyx-se.sh
bash time-nyx-omp.sh
```
### Step 4.4: Compress and decompress the Miranda dataset using our STZ, SPERR, SZ3, ZFP, and MGARD in serial and OpenMP modes, and evaluate the speed (22 minutes).
```
bash time-mrd-se.sh
bash time-mrd-omp.sh
```

### Step 5.1: Compress the Miranda dataset using our STZ, and progressively decompress it at different resolutions. Then, use random access to decompress a 3D ROI box and a 2D ROI slice (1 minute).
```
bash stream.sh 
```
### Step 5.2: Evaluate the decompression time at different resolutions.
```
python3 readprg.py
```
### Step 5.3: Evaluate the time breakdown for decompressing all data, randomly accessing a 3D ROI box, and a 2D slice.
```
python3 readrd.py 
```

### Baseline compressors: [SZ3](https://github.com/szcompressor/SZ3.git); [ZFP](https://github.com/LLNL/zfp); [SPERR](https://github.com/NCAR/SPERR); [MGARDX](https://github.com/CODARcode/MGARD/tree/master).

