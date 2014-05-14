# Metro Maps of Photos
## April Shen (april@cs.uw.edu)

### Current Version (main.py)

* New focus on social graph, relationships over time
* More lightweight notion of coherence (using face cluster counts and visual smoothness)
* More work on the user interface and interaction components

#### To-do's
* better visual features, for visual continuity (both metro map and social graph)
* better global optimization, especially for structure and coherence
* soft clustering/overlapping community detection of faces (BigClam?)
* more principled (but still fast) coherence algorithm
* coordinated selection between metro map and social graph
* add the geographic map
* zooming and on-the-fly map creation

#### Metro Map
* features: faces (weighted by region size), timestamp (binned), GPS location (binned)
* features weighted by frequency in dataset
* timestamps corrected by drawing from a distribution defined by valid timestamps of
  photos in the same cluster (something of a hack)
* faces clustered by counting occurrences of face groups and sorting based on size of
  group (small is good) and number of photos covered (large is good)
* choose a face cluster per metro line
* for each line, greedily (based on coverage of faces/times/place) choose photos containing
  at least one face in its cluster, throwing out photos not sufficiently coherent with
  existing line on each iteration
* sort each line by time
* visualization: force-directed layout with custom gravity (x-axis based on time,
  y-axis for separation of lines) and collision detection (so nodes don't overlap)
* hacky things: lines are organized vertically roughly so that lines sharing photos are nearby

#### Social Graph (social_main.py)
* approximation of social graph induced from photos
* nodes colored by cluster membership (social group)
* edges weighted by number of photos the nodes co-occur in (relationship strength)
* one graph per time bin, displaying clusters chosen greedily based on coverage of
  photos
* faces as nodes, chosen (for now) based on size of face

-------------------------------------------------

### Old Version (old_main.py)

Currently (basically full pipeline):
* cover faces (analogous to words) as in metro maps
* also bin timestamps and GPS locations and cover those
* weighted by frequency in dataset
* compute coverage w/ greedy algorithm
* both binary 'bag of faces' + percentage of image taken up by face
* build coherence graph (nodes=short coh. paths, edges if overlap)
* use submodular orienteering to find high-coverage paths
* we now do orienteering before we start choosing paths, since my guess is that
 these paths don't change much on each iteration so recalculating is a waste of time

To do's:
* adding back connectivity
* fancier things with face regions
* really need to speed up submodular orienteering, but how?
* will we need to have connectivity as a post-step?
* will we need to put orienteering back into the loop?
* "activations" i.e. small friend groups?

Visualization -- making the map: [code from ezyang]
* stops 'labeled' with images (or nodes ARE images)
* clicking on a stop brings up larger view of photo and maybe some metadata (faces etc)
* eventually: toggle between lines=events and lines=social circles
* zoom can control granularity of time (place?)
* fancy automatic layout generation
* scrub or hover to show intervening photos on the line