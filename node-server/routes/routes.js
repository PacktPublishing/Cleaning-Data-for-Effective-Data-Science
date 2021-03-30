// import other routes
const userRoutes = require('./users');

const appRouter = (app, fs) => {

    // default route
    app.get('/', (req, res) => {
        res.send('welcome to the development api-server');
    });

    app.get('/disabled', (req, res) => {
        res.setHeader('Content-Type', 'text/plain');
        res.status(410).send('Resource has been disabled');
    });

    // other routes
    userRoutes(app, fs);

};

module.exports = appRouter;
