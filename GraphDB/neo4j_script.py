from neo4j import GraphDatabase
import json
#Indicate to driver where neo4j is running
driver = GraphDatabase.driver("bolt://localhost",auth=("test_user","password"))
#Parse data
with open("./data.json",'r') as file:
    #Need to make a session where you will run all your cypher queries
    with driver.session() as session:
        count = 0
        tx = session.begin_transaction()
        for line in file.readlines(): #can limit lines with [:100] after ()
            if (line == "[\n" or line == "]"):
                continue
            item = json.loads(line[:-2])
            # print(item)
            '''WITH {tweet} AS Tweet
                Merge (a:Tweet{id:$value.id,date:value.date,train_id:value.train_id})
                Merge (d:Date{date:value.date})
                Merge (a)-[:SAME_DATE]->(d)'''
            '''CREATE (a:Tweet{id:$value.id,date:$value.date,train_id:$value.train_id})'''
            tx.run('''start n=node(*) return n''',
                   parameters={'tweet': item}, value=item) #add paramets here
            #Batch processing to run 1000 tweets as a time as these commits are quite time intensive
            count += 1
            tx.commit()
            break   # UNCOMMENT IF PUSHING MULTIPLE ROWS/NODES OF DATA 
            if count > 1000:
                tx.commit()
                tx = session.begin_transaction()
                count = 0
        tx.commit()
# import json
# with open("./data.json", "r") as read_file:
#     for line in read_file.readlines():
#         print(line)
#         if (line == "[\n" or line == "]"):
#             continue
#         item = json.loads(line[:-2])
#         print(item)
