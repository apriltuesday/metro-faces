// Social graph code
function socialFaces(filename, index, size, color, container) {

    var me = "April Shen";
    var width = size,
	height = size,
	padding = 20,
	circlePadding = 20,
	rad = 30;

    var force = d3.layout.force()
	.charge(-200)
	.linkDistance(function(d) { return 100*Math.pow(d.value, -1.5); })
	.size([width, height]);

    var zoom = d3.behavior.zoom()
	.scaleExtent([1, 10])
	.on("zoom", zoomed);

    var svg = container.append("svg")
	.attr("width", width)
	.attr("height", height)
	.on("click", function() { d3.select("#show-image").html(""); })
	.call(zoom);

    var g = svg.append("g");

    function zoomed() {
	g.selectAll("circle")
	    .style("stroke-width", 5 / d3.event.scale + "px")
	    .attr("r", rad / d3.event.scale + "px");
	g.selectAll("image")
	    .attr("x", -rad / d3.event.scale + "px")
	    .attr("y", -rad / d3.event.scale + "px")
	    .attr("width", 2*rad / d3.event.scale + "px")
	    .attr("height", 2*rad / d3.event.scale + "px");
	g.selectAll("line")
	    .style("stroke-width", function(d) { return Math.sqrt(d.value) / d3.event.scale; });
	g.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
    }

    d3.json(filename, function(error, graph) {
	    force
		.nodes(graph.nodes)
		.links(graph.links)
		.start();

	    var link = g.selectAll(".link")
		.data(graph.links)
		.enter().append("line")
		.attr("class", "link")
		.style("stroke-width", function(d) { return Math.sqrt(d.value); });

	    var node = g.selectAll(".node")
		.data(graph.nodes)
		.enter().append("g");

	    node.append("circle")
		.attr("r", rad)
		.style("fill", "white")
		.style("stroke", function(d) { return color(d.group[0]); })
		.style("stroke-width", 5);	        

	    node.append("image")
		.attr("xlink:href", function(d) {
			return "images/" + d.name + index + ".png"
		    })
		.attr("x", -rad)
		.attr("y", -rad)
		.attr("width", 2*rad)
		.attr("height", 2*rad);

	    node.append("title")
		.text(function(d) { return d.name; });

	    // create name strings to display groups
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
			d3.select("#show-names" + index)
			    .html(names[d.group[0]]);
		    }
		});

	    force.on("tick", function(e) {
		    node.each(gravity(.2 * e.alpha))
			.each(collide(0.5));
		    link.attr("x1", function(d) { return d.source.x; })
			.attr("y1", function(d) { return d.source.y; })
			.attr("x2", function(d) { return d.target.x; })
			.attr("y2", function(d) { return d.target.y; });

		    node.attr("transform", function(d) {
			    return "translate(" + d.x + "," + d.y + ")";
			});
		});
	});

    // Draw ME towards center
    function gravity(alpha) {
	return function(d) {
	    if (d.name == me) {
		d.x = width / 2.0;
		d.y = height / 2.0;
	    }
	};
    }

    // Resolves collisions between d and all other circles.
    function collide(alpha) {
	var quadtree = d3.geom.quadtree(force.nodes());
	return function(d) {
	    var r = rad + padding,
		nx1 = d.x - r,
		nx2 = d.x + r,
		ny1 = d.y - r,
		ny2 = d.y + r;
	    quadtree.visit(function(quad, x1, y1, x2, y2) {
		    if (quad.point && (quad.point !== d)) {
			var x = d.x - quad.point.x,
			    y = d.y - quad.point.y,
			    l = Math.sqrt(x * x + y * y),
			    r = rad + rad + circlePadding
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