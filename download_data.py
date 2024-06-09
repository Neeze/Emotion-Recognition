import gdown
import zipfile

url = "https://drive.google.com/file/d/17MduW7Ec2h2mBXCGB0nz4OIPXoo7AUHG/view?usp=drive_link"
output = "archive.zip"
gdown.download(url=url, output=output, fuzzy=True)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('data')