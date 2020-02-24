from neo4j.v1 import GraphDatabase
import json
#Indicate to driver where neo4j is running
driver = GraphDatabase.driver("bolt://localhost",auth=("neo4j","neo"))
#Parse data
with open("dataset/capstone_test_data.json",'r') as file
    #Need to make a session where you will run all your cypher queries
    with driver.session() as session:
        count = 0
        tx = session.begin_transaction()
        for line in file.readlines(): #can limit lines with [:100] after ()
            item = json.loads(line)
            # print(item)

            tx.run('''
                WITH {tweet} AS Tweet
                Merge (a:Tweet{id:value.id,date:value.date,train_id:value.train_id})
                Merge (d:Date{date:value.date})
                Merge (a)-[:SAME_DATE]->(d)
            ''', parameters={'tweet': item}) #add paramets here
            #Batch processing to run 1000 tweets as a time as these commits are quite time intensive
            count += 1
            if count > 1000:
                tx.commit()
                tx = session.begin_transaction()
                count = 0
        tx.commit()
