sudo apt-get update
pip install --upgrade pip setuptools wheel packaging
pip3 install tensorrt_llm
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_5/NsightSystems-linux-cli-public-2025.5.1.121-3638078.deb
sudo dpkg -i NsightSystems-linux-cli-public-2025.5.1.121-3638078.deb
pip install cuda-python==12.8 cuda-core
pip install numpy==1.23.0