import launch

if not launch.is_installed("memory-tempfile"):
    launch.run_pip("install memory-tempfile", "requirements for autoMBW")
