# Node API Server Starter Kit
This starter kit contains everything you'll need to create your very own Node-based API server, using Express JS and configured to serve JSON files for data. 

## Why an API server that serves JSON files?

Driven, as always, by the need to solve a problem I had on how to create a dummy API server that could return various bits of data in a realistic fashion (thus, removing the need to hard code everything), I went down the path of creating a dual project: one for the main React app I was using, with an attached API server using Express JS (i.e. this project).



## Using the API server

I'll run over the basics here to get you up and running, but I wrote an [accompanying article](https://robkendal.co.uk/blog/how-to-build-a-restful-node-js-api-server-using-json-files/) on this very tool that is much more comprehensive and covers many more use cases. You can [read the full Node-based API server article](https://robkendal.co.uk/blog/how-to-build-a-restful-node-js-api-server-using-json-files/) on my website.

### Getting started

First things first, you'll need to fork or clone this repository, and run the install command of your choosing (preferrably Yarn):

```
yarn install

// or

npm install
```
And that's really about it (see, I said it was simple!). To fire up the server and have it do stuff, you'll need to start it with the familiar command:

```
yarn start

// or

npm start
```

### Accessing the server and returning data

The server should be running by now, and you can visit `http://localhost:3001` to see it in action. 

By default, it doesn't return a great deal, but if you visit `http://localhost:3001/users` -- which will automatically issue a GET request to our running API server -- you'll see a simple JSON object populated with some dummy user data.

## Expanding the server

This starter kit is really designed as a kick off point for your own API adventures. If you would like to extend the functionality for your own purposes then you need to do these three things:

1. Add a new JSON file with your relevant data to the main data entry point for the project, `./data`
2. Add a route file that will access this data into `./routes/[your route file].js` -- hint: use the `./routes/users.js` as a starting point
3. Add your new route file into the main routes file located at `./routes/routes.js`






