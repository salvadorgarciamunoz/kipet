from kipet.utils.data_tools import *
import matplotlib.pyplot as plt

def translate_weifengs(d_filename,time_index_filename):
    idx_to_time = dict()
    f = open(time_index_filename,'r')
    for line in f:
        if line not in ['','\t','\n','\t\n',' \n']:
            l = line.split()
            idx = int(l[0])
            time = float(l[1])
            idx_to_time[idx] = time

    f.close()

    tuple_to_data = dict()
    f = open(d_filename,'r')
    for line in f:
        if line not in ['','\t','\n','\t\n',' \n']:
            l = line.split()
            i = int(l[0])
            j = int(l[1])
            d = float(l[2])
            t = idx_to_time[i]
            tuple_to_data[t,j] = d
            
    f.close()
    f = open("tmp",'w')
    for k,v in tuple_to_data.iteritems():
        f.write("{0} {1} {2}\n".format(k[0],k[1],v))
    f.close()
    
    D_frame = read_spectral_data_from_txt("tmp")
    write_spectral_data_to_txt("Dij_case51.txt",D_frame)
    write_spectral_data_to_csv("Dij_case51.csv",D_frame)

translate_weifengs("D.init","51a.idx")

