// Metro map code
function metro(filename, color, container) {

    var width = 1500,
	height = 1000,
	padding = 20,
	circlePadding = 20,
	rad = 5, //radius of metro stop
	wid = 10; //width of metro line
    var numLines = 20;
    var numPlaces = 200;
    var domain = [];
    for (var i = 0; i < numLines; i++)
	domain.push(i);

    // Scales; domains get set according to data in map
    var time = d3.time.scale()
	.range([2*padding, width-2*padding]);
    var yScale = d3.scale.ordinal()
	.domain(domain) //color.domain())
	.rangeBands([2*padding, height-2*padding]);
    // Current "center of gravity" on y-axis for each line
    // This gets re-evaluated on each tick
    var centers = [];
    var counts = []; //number of nodes in each bin
    for (var i = 0; i< numLines; i++) {
	centers.push(yScale(i));
	counts.push(0);
    }


    var axis; //init when we make the map
    var force = d3.layout.force()
	.charge(-100)
	.friction(0.5)
	//	.linkStrength(0.3)
	.size([width, height]);

    var svg = container.append("svg")
	.attr("width", width)
	.attr("height", height)
	.on("click", function() { d3.select("#show-image").html(""); });

    var zoom = d3.behavior.zoom()
	.scaleExtent([1, 10])
	.on("zoom", zoomed);
    svg.call(zoom); //TODO need to zoom axis also

    var g = svg.append("g");
    
    function zoomed() {
	g.selectAll("circle")
	    .attr("r", rad / d3.event.scale + "px")
	    .style("stroke-width", 1.5 / d3.event.scale + "px");
	g.selectAll("line")
	    .style("stroke-width", wid / d3.event.scale + "px");
	g.attr("transform", "translate(" + d3.event.translate + ")" +
	       "scale(" + d3.event.scale + ")");
	//		redrawAxis();
    }

    // make map
    d3.json(filename, function(error, graph) {
	    if (error) console.warn(error);

	    for (var i=0; i<graph.nodes.length; i++) {
		var v = graph.nodes[i];
		v.time = new Date(v.time * 1000); //s->ms conversion
		if (v.time.getFullYear() < 2000)
		    v.time.setFullYear(2010); //something random
	    }
	    
	    force
		.nodes(graph.nodes)
		.links(graph.links);
	    force.links().reverse(); //need to draw in reverse order...
	    
	    time.domain(d3.extent(force.nodes(), function(d) { return d.time; }));
	    axis = d3.svg.axis().scale(time)
		.orient("bottom")
		.ticks(d3.time.years, 1)
		.tickFormat(d3.time.format('%Y'));  //%b?
	    svg.append("g")
		.attr("class", "axis")
		.attr("transform", "translate(" + padding + ", " + (height-2*padding) + ")")
		.call(axis);

	    // Initialize positions
	    force.nodes().forEach(function(d, i) {
		    d.y = yScale(d.line[0]);
		    counts[d.line[0]-1]++;
		});
	    force.start();
	   
	    var link = g.selectAll(".link")
		.data(force.links())
		.enter().append("line")
		.attr("class", "link")
		.style("stroke", function(d) { return d.line<=numLines ? color(d.line) : "#CCCCCC"; }) //TODO overlapping clusters
		.style("stroke-width", wid)
		.style("stroke-opacity", function(d) {return d.line<=numLines? 1:0.7;});

	    var node = g.selectAll(".node")
		.data(force.nodes())
		.enter().append("circle")
		.attr("class", "node")
		.attr("r", rad)
		.style("opacity", function(d) {return d.line[0]<=numLines? 1:0.7;})
		.style("fill", function(d) { return d.line[0]<=numLines ? color(d.line[0]) : "#CCCCCC"; });

	    node.append("title")
		.text(function(d) { return d.id; });

	    node.on("mouseover", function(d) {
		    if (d) {
			d3.select("#show-image")
			    .html("<img style=\"max-width:600px;max-height:600px;\"" +
				  "src=\"images/" + d.id + ".png\" >");
		    }
		});
	    /*
	    node.on("click", function(d) {
		    if (d) {
			var faces = d.faces; //only include faces in this photo
			var longs = []; // don't restrict these (will happen in geo map)
			var lats = [];

			// Create a date window (including conversion to unix time)
			var t = d.time.getTime() / 1000;
			var MONTH = 2.63e6; // number of seconds in a month
			var times = [t - 12*MONTH, t + 12*MONTH];

			//main.mapFromParams('zoom', faces, times, longs, lats);
			// TODO now what???? reload?
		    }
		});
	    */
	    force.on("tick", function(e) {
		    // TODO recompute centers
		    node.each(gravity(e.alpha))
			.each(collide(0.5))
			//			.each(function(d) {d.py
			.attr("cx", function(d) { return d.x; })
			.attr("cy", function(d) { return d.y = Math.max(2*padding, Math.min(height - 2*padding, d.y)); });

		    link.attr("x1", function(d) { return d.source.x; })
			.attr("y1", function(d) { return d.source.y; })
			.attr("x2", function(d) { return d.target.x; })
			.attr("y2", function(d) { return d.target.y; });
		});
	});

    // Draw axis after a zoom
    function redrawAxis() {
	var visible = force.nodes().filter(function(d) {
		return d.x > padding && d.x < width-padding
		&& d.y > padding && d.y < height-padding; });
	time.domain(d3.extent(visible, function(d) { return d.time; }));
	svg.call(axis);
    }
	
    // Move nodes toward cluster focus, separated vertically by line id
    // and horizontally by time
    function gravity(alpha) {
	return function(d) {
	    //d.y += (centers[d.line[0]] - d.y) * alpha;
	    d.y = yScale(d.line[0]);	    
	    d.x += (time(d.time) - d.x) * alpha;
	    //d.x = time(d.time);
	};
    }

    // Resolves collisions between d and all other circles.
    function collide(alpha) {
	var quadtree = d3.geom.quadtree(force.nodes());
	return function(d) {
	    var r = d.radius + padding,
		nx1 = d.x - r,
		nx2 = d.x + r,
		ny1 = d.y - r,
		ny2 = d.y + r;
	    quadtree.visit(function(quad, x1, y1, x2, y2) {
		    if (quad.point && (quad.point !== d)) {
			var x = d.x - quad.point.x,
			    y = d.y - quad.point.y,
			    l = Math.sqrt(x * x + y * y),
			    r = d.radius + quad.point.radius + circlePadding
			if (l < r) {
			    l = (l - r) / l * alpha;
			    d.x -= x *= l;
			    d.y -= y *= l;
			    quad.point.x += x;
			    quad.point.y += y;
			}
		    }
		    return x1 > nx2 || x2 < nx1 || y1 > ny2 || y2 < ny1;
		});
	};
    }

}
