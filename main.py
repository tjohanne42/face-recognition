from website import create_app

app = create_app()

if __name__ == "__main__":
	app.run(debug=True, threaded=True)

# tuto flask
#https://www.youtube.com/watch?v=dam0GPOAvVI&t=2183s