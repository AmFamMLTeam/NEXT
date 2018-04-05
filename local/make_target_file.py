import argparse
import sys
import os
from pathlib2 import Path
import yaml


PUBLIC = os.path.join(os.path.dirname(__file__), '..', 'public')


parser = argparse.ArgumentParser()
parser.add_argument('dir', default='')
parser.add_argument('out', default='targets.yml')
args = parser.parse_args(sys.argv[1:])

search_dir = os.path.join(PUBLIC, args.dir)


def make_target(p):
    p = str(p)
    p = p.split('/public/')[-1]
    t = {'image': p}
    return t


paths = Path(search_dir).glob('**/*')

paths = filter(lambda p: p.is_file(), paths)

targets = list(map(make_target, paths))

with open(args.out, 'w') as f:
    yaml.dump(targets, f)






