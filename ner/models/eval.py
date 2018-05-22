import subprocess
import sys

def conlleval(label_predict, label_path, metric_path):
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)
    command='perl conlleval_rev.pl < '+label_path
    results=subprocess.check_output(command, shell=True, cwd='D:\E-medical records analysis')
    fr=open(metric_path, 'w')
    sys.stdout=fr
    print(results.decode('utf-8'))
