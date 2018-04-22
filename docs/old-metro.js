// Metro map code
function metro(filename, color, faces, dates, longs, lats, container) {

    var width = 1000,
	height = 600,
	padding = 10,
	circlePadding = 5,
	rad = 8, //radius of metro stop
	wid = 10; //width of metro line

    // Scales; domains get set according to data in map
    var time = d3.time.scale()
	.range([padding, width-padding]);
    var yScale = d3.scale.ordinal()
	.domain(color.domain())
	.rangeBands([2*padding, height-2*padding]);

    var axis; //init when we make the map
    var force = d3.layout.force()
	.charge(-200)
	.linkDistance(50)
	.linkStrength(function(d) { return d.line == 0 ? 0.1 : 0.3; })
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
	    .style("stroke-width", function(d) { return (d.line == 0 ? wid / 2 / d3.event.scale : wid / d3.event.scale) + "px"; });
	g.attr("transform", "translate(" + d3.event.translate + ")" +
	       "scale(" + d3.event.scale + ")");
	//		redrawAxis();
    }

    function dragstart(d) {
	d3.select(this).classed("fixed", d.fixed = true);
    }

    // make map
    d3.json(filename, function(error, graph) {
	    if (error) console.warn(error);
	    
	    force
		.nodes(graph.nodes)
		.links(graph.links)
		.start();
	    
	    time.domain(d3.extent(force.nodes(), function(d) { return dates[d.id]; }));
	    axis = d3.svg.axis().scale(time)
		.orient("bottom")
		.ticks(d3.time.years, 1)
		.tickFormat(d3.time.format('%Y')); 
	    svg.append("g")
		.attr("class", "axis")
		.attr("transform", "translate(" + padding + ", " + (height-2*padding) + ")")
		.call(axis);
	   
	    var link = g.selectAll(".link")
		.data(graph.links)
		.enter().append("line")
		.attr("class", "link")
		.style("stroke", function(d) { return color(d.line); })
		.style("stroke-width", wid)
		.style("stroke-opacity", 1);

	    link.filter(function(d) { return d.line == 0; })
		.style("stroke-width", wid/2)
		.style("stroke-opacity", .6)
		.style("stroke-dasharray", "4, 4");

	    var node = g.selectAll(".node")
		.data(graph.nodes)
		.enter().append("circle")
		.attr("class", "node")
		.attr("r", rad)
		.style("fill", "white")
		.style("stroke", "black");

	    node.append("title")
		.text(function(d) { return d.id; });

	    node.on("mouseover", function(d) {
		    if (d) {
			d3.select("#show-image")
			    .html("<img style=\"max-width:600px;max-height:600px;\"" +
				  "src=\"images/" + d.id + ".png\" >");
		    }
		});

	    var drag = force.drag()
		.on("dragstart", dragstart);
	    node.call(drag);
		
		      
	    force.on("tick", function(e) {
		    node.each(gravity(.2 * e.alpha))
			.each(collide(0.9))
			.attr("cx", function(d) { return d.x; })
			.attr("cy", function(d) { return d.y = Math.max(padding, Math.min(height - padding, d.y)); });

		    // TODO links not colliding?
		    // one possibility: faking nodes
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
	time.domain(d3.extent(visible, function(d) { return dates[d.id]; }));
	svg.call(axis);
    }
	
    // Move nodes toward cluster focus, separated vertically by line id
    // and horizontally by time
    function gravity(alpha) {
	return function(d) {
	    d.y += (yScale(d.line) - d.y) * alpha;
	    d.x = time(dates[d.id]);
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