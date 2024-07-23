var canvas = document.querySelector('#paint');
var ctx = canvas.getContext('2d');
(function() {
	
	
	
	var sketch = document.querySelector('#sketch');
	var sketch_style = getComputedStyle(sketch);
	canvas.width = parseInt(sketch_style.getPropertyValue('width'));
	canvas.height = parseInt(sketch_style.getPropertyValue('height'));
	ctx.fillStyle = "white";
	ctx.fillRect(0, 0, canvas.width, canvas.height);

	var mouse = {x: 0, y: 0};
	var last_mouse = {x: 0, y: 0};
	
	/* Mouse Capturing Work */
	canvas.addEventListener('mousemove', function(e) {
		last_mouse.x = mouse.x;
		last_mouse.y = mouse.y;
		
		mouse.x = e.pageX - this.offsetLeft;
		mouse.y = e.pageY - this.offsetTop;
	}, false);
	
	
	/* Drawing on Paint App */
	ctx.lineWidth = 5;
	ctx.lineJoin = 'round';
	ctx.lineCap = 'round';
	ctx.strokeStyle = 'black';
	
	canvas.addEventListener('mousedown', function(e) {
		canvas.addEventListener('mousemove', onPaint, false);
	}, false);
	
	canvas.addEventListener('mouseup', function() {
		canvas.removeEventListener('mousemove', onPaint, false);
        sendImage(canvas);
       
	}, false);
	
	var onPaint = function() {
		ctx.beginPath();
		ctx.moveTo(last_mouse.x, last_mouse.y);
		ctx.lineTo(mouse.x, mouse.y);
		ctx.closePath();
		ctx.stroke();
        
	};
	
}());


function sendImage(canvas) {

	var canvasData = canvas.toDataURL("image/jpeg");
    let formData = new FormData(); 
    formData.append("image", canvasData);
    response = fetch('api/v1/infer_doodle', {
      method: "POST", 
      body: formData
    }).then(response => {
		response.json().then((data) => {
			console.log(data);
			document.getElementById("infer_results").textContent=data.result;
		});
	});
	
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	ctx.fillStyle = "white";
	ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
