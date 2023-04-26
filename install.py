import launch

if not launch.is_installed("memory-tempfile"):
    launch.run_pip("install memory-tempfile", "requirements for autoMBW")
if not launch.is_installed("image-reward"):
    launch.run_pip("install git+https://github.com/Oyaxira/ImageReward.git@main", "requirements for ImageReward")
