const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const mysql = require('mysql');
const testTweets = require('./testTweets.json');
app.set('view engine', 'pug');
app.use(express.static(__dirname + '/public'));
app.use(bodyParser.json());
var first = 0;
const connection = mysql.createConnection({
  user:'admin',
  password:'Private2712!',
  host:'database-1.cok63qqiofsd.us-east-1.rds.amazonaws.com',
  database:'data',
  multipleStatements: true
});


connection.connect((err) => {
  if(err){
    console.log('Error connecting to Database...');
    return;
  }
  console.log('Connected to Database!');
});



app.get('/', (req, res) => {
  connection.query("START TRANSACTION; SELECT * FROM data.train_data ORDER BY train_id ASC LIMIT 1; DELETE FROM data.train_data ORDER BY train_id ASC LIMIT 1; SELECT COUNT(*) FROM data.classified_tweets WHERE relevant = 001; SELECT COUNT(*) FROM data.classified_tweets WHERE relevant = 000; COMMIT;",(err,rows) => {
    if(err){
      console.log('Error selecting tweets from Database...');
      return;
    }
    console.log('Successfully recieved tweets from database');
    if(first == 0){
      first = 1;
      res.render('index', {
        title: 'Tweet Labeler',
        data: rows[1],
        relevant_count: rows[3][0]['COUNT(*)'],
        irrelevant_count: rows[4][0]['COUNT(*)']
      });
    }
    else{
      res.send({ndata:rows[1],rc:rows[3][0]['COUNT(*)'],ic:rows[4][0]['COUNT(*)']});
    }
  });

});
app.post('/',(req,res) => {
  var newRow = {
    relevant: req.body.relevant,
    created_at: req.body.created_at,
    date: req.body.date,
    time: req.body.time,
    timezone: req.body.timezone,
    user_id: req.body.user_id,
    username: req.body.username,
    name: req.body.name,
    place: req.body.place,
    tweet: req.body.tweet,
    mentions:req.body.mentions,
    urls: req.body.urls,
    replies_count: req.body.replies_count,
    retweets_count: req.body.retweets_count,
    likes_count: req.body.likes_count,
    hashtags: req.body.hashtags,
    cashtags:req.body.cashtags,
    link: req.body.link,
    retweet: req.body.retweet,
    video: req.body.video,
    near: req.body.near,
    geo: req.body.geo,
    source: req.body.source,
    user_rt_id: req.body.user_rt_id,
    user_rt: req.body.user_rt,
    retweet_id: req.body.retweet_id,
    reply_to: req.body.reply_to,
    retweet_date: req.body.retweet_date,
    translate:req.body.translate,
    trans_src:req.body.trans_src,
    trans_dest: req.body.trans_dest,
    photos: req.body.photos,
    conversation_id:req.body.conversation_id,
    id:req.body.id
  };
  connection.query('INSERT INTO data.classified_tweets SET ?',newRow,(err,res) => {
    if(err){
      console.log("Failed to insert new tweet into database...");
      var reRow = {
        created_at: req.body.created_at,
        date: req.body.date,
        time: req.body.time,
        timezone: req.body.timezone,
        user_id: req.body.user_id,
        username: req.body.username,
        name: req.body.name,
        place: req.body.place,
        tweet: req.body.tweet,
        mentions:req.body.mentions,
        urls: req.body.urls,
        replies_count: req.body.replies_count,
        retweets_count: req.body.retweets_count,
        likes_count: req.body.likes_count,
        hashtags: req.body.hashtags,
        cashtags:req.body.cashtags,
        link: req.body.link,
        retweet: req.body.retweet,
        video: req.body.video,
        near: req.body.near,
        geo: req.body.geo,
        source: req.body.source,
        user_rt_id: req.body.user_rt_id,
        user_rt: req.body.user_rt,
        retweet_id: req.body.retweet_id,
        reply_to: req.body.reply_to,
        retweet_date: req.body.retweet_date,
        translate:req.body.translate,
        trans_src:req.body.trans_src,
        trans_dest: req.body.trans_dest,
        photos: req.body.photos,
        conversation_id:req.body.conversation_id,
        id:req.body.id
      };
      connection.query('INSERT INTO data.train_data SET ?',reRow, (err,rest) => {
        if(!err){
          console.log("Put the row back in train_data");
        }
      });
      return;
    }
    console.log("Inserted new Tweet Successfully!");
  });
  res.end()
});

const server = app.listen(9600,() => {
   console.log(`Express running → PORT ${server.address().port}`);
});
