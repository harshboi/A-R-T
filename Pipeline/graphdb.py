################################################################################################
# Neo4j test for noun as node (required another db to store tweet)
# Note we only create nodes here and not the relationships
################################################################################################

from neo4j import GraphDatabase
import json

def fetchDriver(username,password):
    #Indicate to driver where neo4j is running
    driver = GraphDatabase.driver("bolt://localhost:7687",auth=(username,password), encrypted=False)
    return driver

def addToGraph(driver,nouns,date,tweet_id):
    # idd = 2
    with driver.session() as session:
        tx = session.begin_transaction()
        count = 0
        idd = tweet_id
#         tx.run('''Match (a:Tweet {type: $word}) return a.num''', word=noun)
        # nouns = nltk_nouns(tweet_data['tweet'])
        for noun in nouns:
            result = tx.run('''Match (a:Tweet {type: $word}) return a''', word=noun.lower())
            result = result.records()
            flag = True
            # date = tweet_data['date']
            # date = "2020-04-14"
#             print(date)
            for record in result:
                flag = False
                find_index = -1
                for i in range ( len( record['a']['last10DaysDate'] )):
                    if record['a']['last10DaysDate'][i] == date:
                        find_index = i
                        break
                if find_index != -1:
                    tx.run('''MATCH (a:Tweet {type:$word}) SET a.num = a.num+1, a.last10DaysCount =
                    a.last10DaysCount[ ..$index ] + (a.last10DaysCount[ $index ] + 1) + a.last10DaysCount[ $index+1.. ] ''', word=noun.lower(), index = find_index)
                else:
                    tx.run('''MATCH (a:Tweet {type:$word}) SET a.num = a.num+1, a.last10DaysDate = a.last10DaysDate + [$date], a.last10DaysCount = a.last10DaysCount + [1]''', word=noun.lower(), date=date)
            if flag:
                tx.run('''CREATE (a:Tweet {type:$word, num:1, id:$idd, last10DaysDate:[$date], last10DaysCount:[1]})''', word=noun.lower(), idd=idd, date=date)
                idd += 1
            tx.run('''Match (a:Tweet), (b:Tweet) where $noun = a.type and $noun = b.type
                        merge (a)-[r: encountred_with {type: $noun}]->(b)''', noun=noun.lower())

        tx.commit()
