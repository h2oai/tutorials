setup-doc:  # Install documentation dependencies
	cd documentation && npm install 

run-doc:  # Run the doc locally
	cd documentation && npm start

update-documentation-infrastructure:
	cd documentation && npm update @h2oai/makersaurus
	cd documentation && npm ls

build-doc-locally:  # Bundles your website into static files for production
	cd documentation && npm run build

serve-doc-locally:  # Serves the built website locally 
	cd documentation && npm run serve