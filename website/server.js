const express = require('express');
const app = express();
const mysql = require('mysql');
const testTweets = require('./testTweets.json');
app.set('view engine', 'pug');
app.use(express.static(__dirname + '/public'));

const connection = mysql.createConnection({
  user:'admin',
  password:'Private2712!',
  host:'database-1.cok63qqiofsd.us-east-1.rds.amazonaws.com',
  database:'data'
});

connection.connect((err) => {
  if(err){
    console.log('Error connecting to Database...');
    return;
  }
  console.log('Connected to Database!');
});



app.get('/', (req, res) => {
  res.render('index', {
    title: 'Tweet Labeler',
    data: testTweets.tweets
  });
});


const server = app.listen(9000,() => {
   console.log(`Express running → PORT ${server.address().port}`);
});
