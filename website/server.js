const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const mysql = require('mysql');
const testTweets = require('./testTweets.json');
app.set('view engine', 'pug');
app.use(express.static(__dirname + '/public'));
app.use(bodyParser.json());

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
  connection.query('SELECT * FROM data.train_data',(err,rows) => {
    if(err){
      console.log('Error selecting tweets from Database...');
      return;
    }
    console.log('Successfully recieved tweets from database');
     res.render('index', {
       title: 'Tweet Labeler',
       data: rows
     });
  });


});

app.post('/',(req,res) => {
  var newRow = {
    tweet: req.body.tweet,
    relevant: req.body.relevant
  };
  connection.query('INSERT INTO data.train_data SET ?',newRow,(err,res) => {
    if(err){
      console.log("Failed to insert new tweet into database...");
      return;
    }
    console.log("Inserted new Tweet Successfully!");
  });
});

const server = app.listen(9000,() => {
   console.log(`Express running → PORT ${server.address().port}`);
});
