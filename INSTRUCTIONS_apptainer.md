# Submit

> TL;DR: 
> * Create a apptainer image with your runtime depedencies 
> * Create scripts for running pion and photon generation

Your finished submission should be a `.tar.gz` file with the following contents: 

```
.
├── container.def
├── container.sif
├── run-photon-sample.sh 
├── run-pion-sample.sh
└── {Runtime Code}/
    └── ...
```

## Create an Apptainer Image
1. Install Apptainer or SingularityCE, or check which one is available on your cluster. 
    - Apptainer: https://apptainer.org/docs/admin/main/installation.html
    - SingularityCE: https://docs.sylabs.io/guides/3.8/admin-guide/installation.html# 

The syntax differences between the two packages are basically none; just replace `singularity` with `apptainer`. In fact, Apptainer should be just a renamed version of Singularity. I also tried running an image generated with Singularity with Apptainer, and it worked.

2. I will show Apptainer commands since it is easier to install, they provide pre-built packages, and it is available on my cluster. Let's define a `.def` file that contains instructions for installing a new container with any additional dependencies.
```
Bootstrap: docker
From: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime


%post
 pip install pandas nflows h5py scikit-learn

```

This script was also used for the CaloChallenge. An explanation of the syntax was provided by Claudius:
- In the header (`Bootstrap` and `From`), we define that we build the image using an existing image (from docker) on which PyTorch 2.7.1 and CUDA are pre-installed. If your code is based on a different version of PyTorch, you can check the available images at https://hub.docker.com/r/pytorch/pytorch/tags. The `runtime` images should be enough, I didn't need the `development` version. Keep in mind that the size of the image will grow if you use the `development` version, and it might affect the loading time, too. Similarly, you can find the TensorFlow images at https://hub.docker.com/r/tensorflow/tensorflow/tags, but I have not tested/used them.  
- In the `%post` section, you define which commands need to be run inside the container when it is created. As you can see, this is where I install the Python packages that I need beyond PyTorch (which was pre-installed in the image we pulled in the header). 
- Note that the container will only provide the environment for the code and other files, like Python scripts or saved network weights, which are still seen from within the container. Therefore, they do not need to be moved inside the container (this is a difference from Docker, in case you are more familiar with that).
- Note also that the creation of the image requires root privileges, so you might run this on your local machine. If there are problems installing packages when the container is built, you can also skip the line in `%post` and install it in the container manually, as in the Mac tutorial below. 
   
3. You can build the image with the command `sudo apptainer build test_container.sif test_file_name.def`. Both Apptainer and SingularityCE allow for building an image without root privileges through the flag `--fakeroot`. I was able to generate one on my cluster with `apptainer build --fakeroot  test_container.sif test_file_name.def`.
4. I propose to use the same format required by the CaloChallenge: a `.tar.gz` file that contains the `.sif` of your environment and all scripts/files that are needed in addition. Please also include the `.def` file.
5. Testing the image (from Claudius):
If you want to test your image, you can do so in the following ways:
- interactive shell within the container: `apptainer shell --nv test_container.sif`. The option `--nv` loads the image with the NVIDIA drivers enabled, so you should see your GPU when you type `nvidia-smi` inside the container. By default, you should see the same filesystem as you saw before entering the container (you should be in the same folder as before). On some machines, you will start back in the home directory and, depending on where you were when you started the Singularity shell, you might not see the right folders. To avoid this, you can mount the current directory to the container by adding `-B $(pwd)` to the call above. 
- just testing a single command within the container: `apptainer exec --nv test_container.sif echo "hello"` will execute `echo "hello"`, i.e. print "hello" from inside the environment and then exit again. The options `--nv` and `-B` are the same as above.

### Install on MacOS
Adapted from CaloChallenge instructions. 

If you want to install Singularity on your MacBook, please follow these steps (Thanks to Ian Pang for working these out!):
1. Install Docker for Apple Silicon (following the instructions here: [https://docs.docker.com/desktop/install/mac-install/](https://docs.docker.com/desktop/install/mac-install/))
2. With Docker, pull the image for `amd64` based OS (e.g., `docker pull ubuntu` to get the Docker image for Ubuntu)
3. To run the `amd64` container, we have to first install an `amd64` emulator with the following commands (same instructions can be found here: [https://enlear.academy/run-amd64-docker-images-on-an-arm-computer-208929004510](https://enlear.academy/run-amd64-docker-images-on-an-arm-computer-208929004510)):
    1. `docker pull tonistiigi/binfmt`
    2. `docker run --privileged --rm tonistiigi/binfmt --install amd64`
4. Now run the `amd64` docker container with command `docker run --rm -ti --platform linux/amd64 my_image_name`
5. Now, follow step 1 at the top to install Go and Singularity.
6. Create your definition file as in step 2, but leave out the section `%post`, as it will lead to errors when you try to install packages now. 
7. Instead, build the `.sif` file now, like in step 3 at the top: by using the command `sudo singularity build image_name.sif definition_file_name.def`
8. To install the missing dependencies in the `.sif` file, copy the `.sif` file to the machine where you run the code for your CaloChallenge submissions. Next, run the command `singularity shell --nv image_name.sif` and then install all the missing dependencies. Doing so should include all the dependencies in the `.sif` file.
9. Finish with step 4 from the top and build the tarball of your submission. 

## Create scripts for photon and pion generation

The submission should contain scripts that show how to run the photon and pion generation (if applicable). They must be self contained and do not require the runner to select different model checkpoints or configurations (to reduce the possiblity of the model being run incorrectly). 

For example (using CaloDiffusion)

```bash
#/bin/bash

if [ $# -lt 3 ]; then
    echo "Error: Missing required arguments" >&2
    echo "Usage: $0 <batch size> <data directory> <output h5>" >&2
    exit 1
fi

BATCHES="$1"
DATA_DIR="$2"
OUTPUT_FILE="$3"

echo "Loading data from: $DATA_DIR"
echo "Output will be saved to: $OUTPUT_FILE"

CONFIG="CaloDiffusion/configs/config_HGCal_photons.json"
CHECKPOINT="CaloDiffusion/checkpoints/checkpoint_HGCal_photons.pth"

RUN_COMMAND="python3 CaloDiffusion/calodiffusion/inference.py --batch-size $BATCHES --config $CONFIG --data-folder $DATA_DIR --hgcal sample -g $OUTPUT_FILE --model-loc $CHECKPOINT"

apptainer exec calodif-test.sif pip3 install -e CaloDiffusion/
apptainer exec calodif-test.sif $RUN_COMMAND

```
The batch size argument is required (it is needed to run compute tests). 

The input and output paths are optional to include as cli args but they must be accessable (e.g. not hardcoded in the codebase itself). 

Please provide instructions in a README.md if anything beyond just running the script is required.  
