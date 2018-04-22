// Social graphmap code
function socialMap(prefix, numTimes, size, color, container) {
    // Single graph, multiple times
    // Connect same CLUSTER w/ lines

    /********* Initialization of force layout ********/

    var me = "April Shen";
    var width = 1500,
	height = 500;

    var zoom = d3.behavior.zoom()
	.scaleExtent([1, 10])
	.on("zoom", zoomed);

    var force = d3.layout.force()
	.charge(-200)
	.linkDistance(function(d) { return d.num == -1? 10 : 5*Math.pow(d.value, -1.5); })
	.size([width, height])
	.on("tick", tick);

    var svg = container.append("svg")
	.attr("width", width)
	.attr("height", height)
	.on("click", function() { d3.select("#show-image").html(""); })
	.call(zoom);

    var g = svg.append("g");

    var nodes = force.nodes(),
	links = force.links();
    var node = g.selectAll(".node"),
	link = g.selectAll(".link");

    var otherI = 0; // XXX how to do this right?
    for (var i = 0; i < numTimes; i++) {
	d3.json(prefix + i + "-graph.json", function(error, graph) {
		if (error) console.error(error);
		var zero = nodes.length;
		var colorChoices = color.range();
		// add this graph's nodes and links
		graph.nodes.forEach(function(element, index, array) {
			var newNode = {num : otherI,
				       id: index,
				       name: element.name,
				       group: element.group,
				       color: colorChoices[element.group[0]] };
			if (newNode.name == me) {
			    newNode.group = [9];
			    newNode.color = colorChoices[9];
			}
			findShared(newNode, colorChoices);
			nodes.push(newNode);
		    });
		graph.links.forEach(function(element, index, array) {
			links.push( {num : otherI,
				    source: zero + element.source,
				    target: zero + element.target,
				    value: element.value,
				    color: '#999' });
		    });

		// restart simulation
		update();
		otherI++;
	    });
    }

    /*********** Main update function ***********/

    function update() {	
	link = link.data(links);
	link.enter().append("line")
	    .attr("class", "link")
	    .style("stroke", function(d) { return d.color; })
	    .style("stroke-width", function(d) { return Math.sqrt(d.value); });

	node = node.data(nodes);
	node.enter().append("circle")
	    .attr("class", "node")
	    .attr("r", 5)
	    .style("fill", function(d) { return d.color; })
	    .append("title")
	    .text(function(d) { return d.name; });
	
	// create name strings to display groups
	/*
	var names = [];
	for (var c in color.domain())
	    names.push("");
	for (var i in force.nodes()) {
	    var theNode = force.nodes()[i];
	    for (var j in theNode.group)
		names[theNode.group[j]] += "<p>" + theNode.name + "</p>";
	}	
	node.on("mouseover", function(d) {
		if (d) {
		    d3.select("#show-names")
			.html(names[d.group[0]]);
		}
	    });
	*/

	force.start();
    }

    /******* Helper functions ********/

    function tick(e) {
	node.each(gravity(.2 * e.alpha));
	link.attr("x1", function(d) { return d.source.x; })
	    .attr("y1", function(d) { return d.source.y; })
	    .attr("x2", function(d) { return d.target.x; })
	    .attr("y2", function(d) { return d.target.y; });
	
	node.attr("cx", function(d) { return d.x; })
	    .attr("cy", function(d) { return d.y; });
    }

    function zoomed() {
	g.selectAll("circle")
	    .style("stroke-width", 1.5 / d3.event.scale + "px")
	    .attr("r", 5 / d3.event.scale + "px");
	g.selectAll("line")
	    .style("stroke-width", function(d) { return Math.sqrt(d.value) / d3.event.scale; });
	g.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
    }

    function gravity(alpha) {
	return function(d) {
	    if (d.name == me) {
		d.x = (width / (2.0*numTimes)) * (2*d.num + 1);
		d.y = height / 2.0;
	    }
	};
    }

    function findShared(newNode, colorChoices) {
	var maxNum = -1;
	var maxEl = null;
	var maxInd = -1;
	nodes.forEach( function(element, index, array) {
		if (newNode.name === element.name) {
		    if (element.num > maxNum && element.num != newNode.num) {
			maxNum = element.num;
			maxEl = element;
			maxInd = index;
		    }
		}
	    });

	if (maxNum > -1) {
	    swap(newNode.color, maxEl.color, colorChoices);
	    newNode.color = maxEl.color;
	    links.push( {num : -1, source: nodes.length, target: maxInd, value: 40, color: newNode.color } );
	}

	//fix the colors
	nodes.forEach( function(el, ind, arr) {
		if (newNode.num === el.num) {
		    el.color = colorChoices[el.group[0]];
		}
	    });
    }

    function swap(x, y, a) {
	var i, j;
	a.forEach( function(el, ind, arr) {
		if (el === x)
		    i = ind;
		if (el === y)
		    j = ind;
	    });
	var temp = a[i];
	a[i] = a[j];
	a[j] = temp;
    }
}