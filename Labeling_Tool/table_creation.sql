CREATE TABLE `train_data` (
  `train_id` int(11) NOT NULL AUTO_INCREMENT,
  `created_at` int(11) DEFAULT NULL,
  `date` date DEFAULT NULL,
  `time` time DEFAULT NULL,
  `timezone` varchar(45) DEFAULT NULL,
  `user_id` int(11) DEFAULT NULL,
  `username` varchar(45) DEFAULT NULL,
  `name` varchar(45) DEFAULT NULL,
  `place` varchar(45) DEFAULT NULL,
  `tweet` varchar(300) DEFAULT NULL,
  `mentions` varchar(45) DEFAULT NULL,
  `urls` varchar(45) DEFAULT NULL,
  `replies_count` int(11) DEFAULT NULL,
  `retweets_count` int(11) DEFAULT NULL,
  `likes_count` int(11) DEFAULT NULL,
  `hashtags` varchar(45) DEFAULT NULL,
  `cashtags` varchar(45) DEFAULT NULL,
  `link` varchar(45) DEFAULT NULL,
  `retweet` tinyint(4) DEFAULT NULL,
  `video` varchar(45) DEFAULT NULL,
  `near` varchar(45) DEFAULT NULL,
  `geo` varchar(45) DEFAULT NULL,
  `source` varchar(45) DEFAULT NULL,
  `user_rt_id` int(11) DEFAULT NULL,
  `user_rt` varchar(450) DEFAULT NULL,
  `retweet_id` int(11) DEFAULT NULL,
  `reply_to` varchar(450) DEFAULT NULL,
  `retweet_date` varchar(45) DEFAULT NULL,
  `translate` varchar(45) DEFAULT NULL,
  `trans_src` varchar(45) DEFAULT NULL,
  `trans_dest` varchar(45) DEFAULT NULL,
  `relevant` tinyint(3) unsigned zerofill DEFAULT NULL,
  `photos` varchar(300) DEFAULT NULL,
  PRIMARY KEY (`train_id`)
) ENGINE=InnoDB AUTO_INCREMENT=34 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
