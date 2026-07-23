#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging

from qsnow.experiments import *
from qsnow.helpers import serialize
from qsnow.interface import *
from qsnow.visualize import *

logging.basicConfig(level=logging.INFO)

# ## Mean = 0.01, Dev = 0.005

# In[ ]:


chip = Chip(15, 15)
d3Tile = SCTile(3)
d5Tile = SCTile(5)

mean = 0.01
dev = 0.005
chip.generate_gaussian_noise(mean, dev)

exp_d3 = SquarePackingExp(chip, d3Tile)
exp_d5 = SquarePackingExp(chip, d5Tile)

# In[ ]:


# chip.show(area_selection_style(chip, {k:chip.loc(k) for k in exp_d3.profile}))
chip.show(area_selection_style(chip, {k: chip.loc(k) for k in exp_d5.profile}))

# In[ ]:


exp_d3.save(
    label="d3-15x15",
    desc="Gaussian Noise (mean=0.01, dev=0.005) with d3 RSC tiles on 15x15 chip",
)
exp_d5.save(
    label="d5-15x15",
    desc="Gaussian Noise (mean=0.01, dev=0.005) with d5 RSC tiles on 15x15 chip",
)

# In[ ]:


exp_d5.run()

# In[ ]:


exp_d3.run()

# In[ ]:


exp_d3.save_results()

# In[ ]:


exp_d5.save_results()

# ## Load/View

# In[ ]:


serialize.summarize_exports("*", "experiments")

# In[ ]:


exp_d3: SquarePackingExp = serialize.import_latest("experiment*d3")
exp_d5: SquarePackingExp = serialize.import_latest("experiment*d5")
d3_results: ExperimentResults = serialize.import_latest("results*d3")
d5_results: ExperimentResults = serialize.import_latest("results*d5")

# In[ ]:


exp_d3.show(d3_results)

# In[ ]:


exp_d5.show(d5_results)
