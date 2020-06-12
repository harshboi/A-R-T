################################################################################################
# Neo4j test for noun as node
# Func also for checking relevance after inserted into db (Returns URL, Noun, Num_occurences)
################################################################################################

from neo4j import GraphDatabase
import json

def fetchDriver(username,password):
    #Indicate to driver where neo4j is running
    driver = GraphDatabase.driver("bolt://localhost:7687",auth=(username,password), encrypted=False)
    return driver

def addToGraph(driver, nouns, date, tweet_id, link, username, name, time):
    nouns  = list(set(nouns))
    with driver.session() as session:
        tx = session.begin_transaction()
        count = 0
        for noun in nouns:
            result = tx.run('''Match (a:Tweet {type: $word}) return a''', word=noun.lower())
            result = result.records()
            flag = True
            for record in result:
                flag = False
                find_index = -1
                for i in range ( len( record['a']['last10DaysDate'] )):
                    if record['a']['last10DaysDate'][i] == date:
                        find_index = i
                        break
                if find_index != -1:
                    tx.run('''MATCH (a:Tweet {type:$word}) SET a.num = a.num+1, a.last10DaysCount =
                    a.last10DaysCount[ ..$index ] + (a.last10DaysCount[ $index ] + 1) + a.last10DaysCount[ $index+1.. ], a.link_id = a.link_id + [ $link_id ] ''', word=noun.lower(), index = find_index, link_id = link + " " + tweet_id)
                else:
                    tx.run('''MATCH (a:Tweet {type:$word}) SET a.num = a.num+1, a.last10DaysDate = a.last10DaysDate + [$date], a.last10DaysCount = a.last10DaysCount + [1], a.link_id = a.link_id + [ $link_id ] ''', word=noun.lower(), date=date, link_id = link + " " + tweet_id)
            if flag:
                tx.run('''CREATE (a:Tweet {type:$word, num:1, last10DaysDate:[$date], last10DaysCount:[1], link_id:[ $link_id ]})''', word=noun.lower(), date=date, link_id = link + " " + tweet_id)
            tx.run('''MERGE (a:Author {username:$username, name:$name})''', username=username.lower(), name = name.lower())
            
        for i in range(len(nouns)):
            tx.run('''
                    Match (a:Author { username: $username }), (b:Tweet {type: $noun})
                    MERGE (a)-[: TWEETED {date: $date, time: $time, link_id: $link_id}]->(b)
                    ''', noun=nouns[i].lower(), date=date, time=time, username=username.lower(), link_id = link + " " + tweet_id)
            
            for j in range(i+1, len(nouns)):
                if nouns[i] == nouns[j]:
                    continue
                tx.run('''
                        Match (a:Tweet { type: $noun_1 }), (b:Tweet {type: $noun_2})
                        MERGE (a)-[: ENCOUNTERED_WITH {type: $noun_2}]->(b)
                        MERGE (a)<-[: ENCOUNTERED_WITH {type: $noun_1}]-(b)
                        ''', noun_1=nouns[i].lower(), noun_2=nouns[j].lower())

        tx.commit()

def check_relevance (driver, noun):
    with driver.session() as session:
        tx = session.begin_transaction()
        result = tx.run('''Match (a:Tweet {type: $word}) return a''', word=noun.lower())
        result = result.records()
        ans = []
        for record in result:
#             print(sorted(zip(record['a']['last10DaysDate'], record['a']['last10DaysCount'])))
            date_and_count = sorted(zip(record['a']['last10DaysDate'], record['a']['last10DaysCount']))
            if (date_and_count[-1][1] > 3):
                link_id =  (*map( lambda x: x.split(" "), record['a']['link_id'] ), ) # Contains all urls encountered for that noun
                for i in range(len(link_id)):
                    print(link_id[i][0].split('/status')[0] + '/status/' + link_id[i][1], noun, date_and_count[-1][1])
#                 print(link_id)
                print("\n")
