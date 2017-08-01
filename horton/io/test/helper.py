from glob import glob

import shutil

from horton import context

filenames = glob("*.py")

# filenames = ["test_cp2k.py"]

for filename in filenames:
    with open(filename) as fh:
        for ln in fh:
            if "get_fn('" in ln:
                fn = ln.split("get_fn('")[1].split("'")[0]
                try:
                    full_path = get_fn('{}'.format(fn))
                    shutil.copyfile(full_path, "cached/{}".format(fn))
                except IOError:
                    print fn
                    pass