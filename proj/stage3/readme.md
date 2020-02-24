##Status

Link to "home improvement" gzip file posted (url.md)

gzipread.py file will directly read the gzip file into a dataframe (requires local copy)

Note: .json file in the archive is not a "strict" json file, so the ordinary pd.read_json call doesn't work 

* We would have to revise the json file structure to get it in this way, easier to go with gzipread program
