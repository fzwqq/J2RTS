import os
jobs_dir = '/seu_share/home/weiweiwu/uRTS/microrts/sub_job'
entries  = os.listdir(jobs_dir)
for file_name in entries:
    print(file_name)
    with open(file_name, "r", encoding='utf-8') as f:
        newline = []
        for line in f.readlines():
            newline.append(line.replace("cd ~/uRTS/microrts", "cd ~/uRTS/microrts/"))
    with open(file_name, "w") as f:
        for line in newline:
            f.writelines(line)
