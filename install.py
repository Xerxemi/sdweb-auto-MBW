import launch

if not launch.is_installed("memory_tempfile"):
    launch.run_pip("install memory-tempfile", "requirements for autoMBW")
if not launch.is_installed("image_reward"):
    launch.run_pip("install git+https://github.com/Oyaxira/ImageReward.git@main", "requirements for ImageReward")
