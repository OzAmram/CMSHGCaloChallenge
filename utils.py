import os
import pickle

import numpy as np


def get_files(field, folder=""):
    print(field, folder)
    if(isinstance(field, list)):
        if(len(folder) > 0): 
            out = [os.path.join(folder, file) if (folder not in file) else file for file in field]
        else: 
            out = field
        return out
    elif(isinstance(field, str)):
        if(not os.path.exists(field)):
            print("File list %s not found" % field)
            return []
        if(".h5" in field): #single file
            if(len(folder) > 0 and folder not in field):
                path = os.path.join(folder, field)
            else:
                path = field
            if(not os.path.exists(path)):
                print("Missing file %s, skipping" % path)
                return []
            return [path]
        with open(field, "r") as f:
            f_list = []
            for line in f:
                line = line.strip()
                if(len(line) == 0): continue
                if(line.lower() == "name"): continue
                if(line.startswith("#")): continue

                if(len(folder) > 0 and folder not in line):
                    path = os.path.join(folder, line)
                else:
                    path = line

                if(path.endswith(".h5") and not os.path.exists(path)):
                    print("Missing file %s, skipping" % path)
                    continue
                f_list.append(path)
            return f_list
    else:
        print("Unrecognized file param ", field)
        return []

def WeightedMean(coord, energies, power=1, axis=-1):
    ec = np.sum(energies * np.power(coord, power), axis=axis)
    sum_energies = np.sum(energies, axis=axis)
    ec = np.ma.divide(ec, sum_energies).filled(0)
    return ec



def GetWidth(mean,mean2):
    width = np.ma.sqrt(mean2-mean**2).filled(0)
    return width

def _separation_power(hist1, hist2, bins):
    """computes the separation power aka triangular discrimination (cf eq. 15 of 2009.03796)
    Note: the definition requires Sum (hist_i) = 1, so if hist1 and hist2 come from
    plt.hist(..., density=True), we need to multiply hist_i by the bin widhts
    """
    hist1, hist2 = hist1 * np.diff(bins), hist2 * np.diff(bins)
    ret = (hist1 - hist2) ** 2
    ret /= hist1 + hist2 + 1e-16
    return 0.5 * ret.sum()

def SetStyle():
    from plotting.plotting_utils import set_cms_style
    set_cms_style()

def LoadJson(file_name):
    import yaml

    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))


# Work around for a dumb pickle behavior...
# https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "HGCalGeo":
            renamed_module = "HGCalShowers.HGCalGeo"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def pickle_load(file_obj):
    return RenameUnpickler(file_obj).load()


def load_geom(geom_filename):
    geom_file = open(geom_filename, "rb")

    geom = pickle_load(geom_file)

    # angle from 0 to 2pi, arctan2 has (y,x) convention for some reason
    geom.theta_map = np.arctan2(geom.xmap, geom.ymap) % (2.0 * np.pi)
    geom.max_ncell = int(round(np.amax(geom.ncells)))
    #print("ncell max",  geom.max_ncell)
    #print("rbins",  np.amax(geom.ring_map) +1)
    return geom
