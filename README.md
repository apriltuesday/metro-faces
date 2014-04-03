April Shen
Metro maps X Photobios

New focus on social graph, relationships
More lightweight notion of coherence (using co-clustering)
More on the user interface and interaction components

-------------------------------------------------

Currently (basically full pipeline):
-cover faces (analogous to words) as in metro maps
-also bin timestamps and GPS locations and cover those
-weighted by frequency in dataset
-compute coverage w/ greedy algorithm
-both binary 'bag of faces' + percentage of image taken up by face
-build coherence graph (nodes=short coh. paths, edges if overlap)
-use submodular orienteering to find high-coverage paths
-we now do orienteering before we start choosing paths, since my guess is that
 these paths don't change much on each iteration so recalculating is a
 waste of time

To do's:
-adding back connectivity
-fancier things with face regions
-really need to speed up submodular orienteering, but how?
-will we need to have connectivity as a post-step?
-will we need to put orienteering back into the loop?
-"activations" i.e. small friend groups?


Visualization -- making the map: [code from ezyang]
-stops 'labeled' with images (or nodes ARE images)
-clicking on a stop brings up larger view of photo and maybe some metadata (faces etc)
-eventually: toggle between lines=events and lines=social circles
-zoom can control granularity of time (place?)
-fancy automatic layout generation
-scrub or hover to show intervening photos on the line