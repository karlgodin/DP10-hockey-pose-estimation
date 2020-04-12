import zipfile
import os

zipped_folder = "C:/Users/karlg/Downloads/SBU"

def main():
    files = []
    for (dirpath, dirnames, filenames) in os.walk(zipped_folder):
        files += [os.path.join(dirpath, file) for file in filenames if file.endswith(".zip")]

    for file in files:
        with zipfile.ZipFile(file, 'r') as zip_file:
            structure = zip_file.namelist()
            txt_paths = [element for element in structure if element.endswith(".txt")]

            for txt_path in txt_paths:
                txt_file = zip_file.read(txt_path)
                with open(txt_path.replace("/", "_"), 'wb') as new_file:
                    new_file.write(txt_file)



if __name__ == "__main__":
    main()