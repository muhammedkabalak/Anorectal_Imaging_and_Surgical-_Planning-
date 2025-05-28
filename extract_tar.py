import tarfile

tar_path = r"C:\Users\memir\Downloads\Task05_Prostate.tar"
extract_path = r"C:\Users\memir\Downloads\prostate_dataset"

with tarfile.open(tar_path) as tar:
    tar.extractall(path=extract_path)

print("✅ Veri başarıyla çıkarıldı.")
