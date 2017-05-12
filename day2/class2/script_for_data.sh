if [ -f ./notMNIST_large.tar.gz ]; then
    echo "Skip to download large dataset.."
else
    wget http://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz
fi

if [ -f ./notMNIST_small.tar.gz ]; then
    echo "Skip to download small dataset.."
else
    wget http://commondatastorage.googleapis.com/books1000/notMNIST_small.tar.gz
fi

if [ -f ./codes.tar.gz ]; then
    echo "Skip to download code.."
else
    wget https://www.dropbox.com/s/yqn09d37sxc1nav/codes.tar.gz
fi

echo "Extract the large dataset.."
tar -xzf notMNIST_large.tar.gz
echo "Extract the small dataset.."
tar -xzf notMNIST_small.tar.gz
echo "Extract the codes.."
tar -xzf codes.tar.gz
