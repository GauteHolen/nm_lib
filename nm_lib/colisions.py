import os
import numpy as np

class CrossSection():

    def __init__(self,dir,species1,species2):
        self.name = f"{species1}-{species2}"
        self.temp,self.sigma = self.get_cs_tempsigma(dir,species1,species2)

    def get_sigma(self,T_ab):
        return np.interp(T_ab,self.temp,self.sigma)*2.80e-21 #m^2


    def find_cs_file(self,dir,species1,species2):
        files = os.listdir(dir)

        col_variations = files.copy()
        
        for i in range(len(files)):
            col_variations[i] = files[i].split("-")[0:2]
            col_variations[i] = col_variations[i][0]+"-"+col_variations[i][-1]

        the_file = ""
        for i,file in enumerate(col_variations):
            if species1+"-"+species2 == file:
                the_file = files[i]
            elif species2+"-"+species1 == file:
                the_file = files[i]
        if the_file =="":
            raise FileNotFoundError(f"Can't find colision file for {species1}-{species2}")
        print(f"Loading cross section data for {species1}-{species2}: {the_file}" )
        return the_file

    def read_cs_file(self,dir,species1,species2):
        filename = self.find_cs_file(dir,species1,species2)
        f = open(dir+"/"+filename, "r")
        f = f.read().split("\n")
        skiprows = 2
        for line in f[:50]:
            if line[0:12] == ";       (eV)":
                break
            else:
                skiprows+=1
                
        N_rows = len(f[skiprows:])
        temp = np.zeros(N_rows-2)
        sigma = np.zeros(N_rows-2)
        for i,line in enumerate(f[skiprows:-2]):
            line = line.split()
            temp[i] = float(line[1])
            sigma[i] = float(line[2])

        return temp,sigma

    def get_cs_tempsigma(self,dir,species1,species2):
        temp,sigma = self.read_cs_file(dir,species1,species2)
        return temp,sigma


if __name__ == "__main__":
    cs = CrossSection("nm_lib/nm_lib/cross-sections","e","h")
    print(cs.get_sigma(4000))