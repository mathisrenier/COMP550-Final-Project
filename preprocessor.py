# COMP 550 FINAL PROJECT

import weka.core.jvm as jvm
import weka.core.packages as packages
from weka.core.classes import complete_classname

jvm.start(packages=True)

help(jvm.start)

jvm.stop()