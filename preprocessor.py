# COMP 550 FINAL PROJECT

import sys
import weka.core.jvm as jvm
import weka.core.packages as packages
from weka.core.classes import complete_classname

def install_package(pkg):
    # install package if necessary
    if not packages.is_installed(pkg):
        print("Installing %s..." % pkg)
        packages.install_package(pkg)
        print("Installed %s, please re-run script!" % pkg)
        jvm.stop()
        sys.exit(0)
    print('Package already installed.')


def remove_package(pkg):
    if packages.is_installed(pkg):
        print("Removing %s..." % pkg)
        packages.uninstall_package(pkg)
        print("Removed %s, please re-run script!" % pkg)
        jvm.stop()
        sys.exit(0)
    print('No such package is installed')

if __name__ == '__main__':
    jvm.start(packages=True)

    install_package('AffectiveTweets')


    jvm.stop()