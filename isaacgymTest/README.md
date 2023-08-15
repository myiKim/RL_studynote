# For Testing IsaacGym Env

This directory is supposed to describe all the required steps to run RL training code using NVIDIA's Isaac Gym (https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_gym_isaac_gym.html)

## Installation (on Windows 10)

For IsaacSim, I followed the most of steps from the official website (https://docs.omniverse.nvidia.com/isaacsim/latest/install_workstation.html)

### Installation of IssacSim App

Go to the official download website (https://www.nvidia.com/en-us/omniverse/) and try "Get Started" and register yourself into NVIDIA Omniverse platform. I've chosen Windows version when downloading the installer. Then, you can basically follow the steps described in here (https://docs.omniverse.nvidia.com/isaacsim/latest/install_workstation.html) to get ready for IssacSim app launching. There should be a few configuration steps you need to go through. You can directly launch the IsaacSim app by going into LIBRARY tab ->  LAUNCH button, but rather I am going to use python environment version, which also launch the app automatically. For more information of python environment, see the page (https://docs.omniverse.nvidia.com/isaacsim/latest/install_python.html)


### Set Python Environment Path for your OS

Once you go to the path you install the IsaacSim on, you can find the packaging folder.
In my case, C:\Users\your-user-name\AppData\Local\ov\pkg\isaac_sim-2022.2.1 gives you the packages 

For Windows 10 users, You can go to Control Panel > System Environment Variables > Properties > Environment Variables and then click the New.. button of the System Variables tab. I set a new path PY_OMNI_PATH as C:\Users\your-user-name\AppData\Local\ov\pkg\isaac_sim-*\python.bat and clicke OK to finish the setting.

Now, when you open a Anaconda Powershell Prompt terminal and type:

```bash
$env:PY_OMNI_PATH 
```
This should give you the right path (the path I mentioned above), if you followed correctly.

### Conda Setup

```bash
conda create -n isaacsym -python=3.8
conda activate isaacsym

```

### Clone the training code

```bash
cd your-working-dir
git clone https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs.git
cd OmniIsaacGymEnvs
& $env:PY_OMNI_PATH -m pip install -e . 
```

This will create a RL environment we need and then install all the relevant packages.

## Run the training code

```bash
cd omniisaacgymenvs
& $env:PY_OMNI_PATH .\scripts\rlgames_train.py task=Ant
```

will lauch the IssacSim backend, and run the training of your policy.


## For more information

Please go to the page for complete tutorial: https://docs.omniverse.nvidia.com/isaacsim/latest
I am still learning how to use Isacc-gym, in general, so whenever I got a cool new stuff , I will try to update this readme as well.