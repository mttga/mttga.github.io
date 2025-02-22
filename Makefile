push_blog: 
	git add . && git commit -m "Update" && git push

build:
	hugo --minify

clean: 
	rm -r public

push_theme:
	cd themes/typo && git add . && git commit -m "$(m)" && git push

run: 
	hugo server --disableFastRender -D