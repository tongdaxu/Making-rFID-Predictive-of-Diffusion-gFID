import turitrove as trove
import os
import zipfile

def mount_imgnet():
    trove.set_auth('tongda_xu', '3956c397-64fb-44bd-8848-c96ac25ea0dd')
    trove.download('dataset/imagenet1k256h5format@1.0.0', '~/data/trove')

    with zipfile.ZipFile("/root/data/trove/imagenet1k256h5format-1.0.0/data/raw.zip") as f:
        f.extractall("/root/data/trove/imagenet1k256h5format-1.0.0/data/raw")

    os.remove("/root/data/trove/imagenet1k256h5format-1.0.0/data/raw.zip")

if __name__ == "__main__":
    mount_imgnet()