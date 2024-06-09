import gdown
import zipfile

url = "https://drive.google.com/file/d/1pbJgllK_zxQr1dF-XOxb5I2klyFzs-fO/view?usp=drive_link"
output = "archive.zip"
gdown.download(url=url, output=output, fuzzy=True)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('data')