const express = require('express');
const app = express();
const testTweets = require('./testTweets.json');
app.set('view engine', 'pug');
app.use(express.static(__dirname + '/public'));


app.get('/', (req, res) => {
  res.render('index', {
    title: 'Tweet Labeler',
    data: testTweets.tweets
  });
});


const server = app.listen(9000,() => {
   console.log(`Express running → PORT ${server.address().port}`);
});
