import sys
import numpy as np
import timeit
import csv
import matplotlib.pyplot as plt
import math
from itertools import permutations
import copy


class Location:
    def __init__(self):
        # nodelist의 키가 node의 id이고 value를 latitude와 longitude의 리스트로 둠
        # 알고리즘에 따라 가장 optimal한 투어의 노드 순서대로 업데이트하여 리스트형태로 보관하는 tourlist
        # 그때마다 갱신되는 optimal 투어의 거리 distance
        self.tourlist = []
        self.nodelist = {}
        self.distance = 10000000000000000000000000 * 1000000000000000000000000

    
    # 데이터를 읽는 Method, sys.argv형태로 command창에서 입력을 받은 readingmaterial을 읽어서 딕셔너리 형태로 node id, 위도,경도를 저장
    def readdata(self,readingmaterial):
        f = open(readingmaterial,'r',encoding = "CP949")
        data = csv.reader(f)
        next(data)
        for row in data:
            self.nodelist[int(row[0])] = [float(row[1]),float(row[2])]
        f.close

# 위도와 경도를 받아서 두 지점사이의 거리를 계산하는 함수
def calDistance(lat1, lon1, lat2, lon2):
    radius = 6371.0

    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)

    a = math.sin(dLat / 2.) * math.sin(dLat / 2.) + math.sin(dLon / 2.)*math.sin(dLon / 2.) * math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1. - a))

    dDistance = radius * c

    return dDistance




if __name__ == "__main__":
    try:
        if len(sys.argv) == 4:

                # command 창에서 input file이름, 알고리즘 종류, outputfile이름을 입력하도록 하세요. 3개 다 입력하지 않으면 에러처리
                # 입력된 input file을 읽어서 Location 클래스 멤버로 두기
                # 입력파일 이름을 reading material로 두기
            readingmaterial = sys.argv[1]
            myTSP = Location()
            myTSP.readdata(readingmaterial)
            dlist = np.zeros((len(myTSP.nodelist)+1,len(myTSP.nodelist)+1)) # matrix의 원소 a(ij)는 i번째노드와 j번째 노드사이의 거리 
            totaldis = 0 # 총투어의 길이

            # 투어의 total 거리를 totaldis, 임의의 두 노드 사이의 거리를 dis2라고 할 때
            # Permutation 알고리즘 사용하세요
            if int(sys.argv[2]) == 1:
                start = timeit.default_timer() 

                for i in range(1,len(myTSP.nodelist)+1):
                    for j in range(i+1,len(myTSP.nodelist)+1):
                        dlist[i,j] = calDistance(myTSP.nodelist[i][0], myTSP.nodelist[i][1], myTSP.nodelist[j][0], myTSP.nodelist[j][1])
                        # i에서 j로 가는것과, j에서 i로 가는것은 같으므로(Symmetric tour)
                        dlist[j,i] = dlist[i,j]


                # Key값인 노드id값들의 순열을 튜플형태로 하는 permutation(myTSP.nodelist) e.g. (1,2,3,4,5,6,7,8,9,10), (1,2,3,4,5,6,7,8,10,9)....

                # 1을 제외하고 순열을 돌릴 것이므로 1을 제외한 임의의 리스트를 만듬
                nlist = list(myTSP.nodelist.keys())
                del nlist[0]

                for permut in permutations(nlist):
                    permut1 = list(permut)
                    permut1.insert(0,1) # 맨 앞에 1을 추가한 리스트를 만듬
                    # 이웃한 노드끼리 거리를 계산해서 총 distance에 더해라    
                    # dis2를 두 노드사이의 거리라고 할 때 다이나믹 프로그래밍 방식으로 dlist matrix에 저장하면서 꺼내쓰기
                    for i in range(len(permut1)-1):
                        totaldis += dlist[permut1[i],permut1[i+1]]
                    # 마지막 노드에서 처음노드 사이의 거리도 totaldis에 추가해라
                    totaldis += calDistance(myTSP.nodelist[permut1[0]][0],myTSP.nodelist[permut1[0]][1],myTSP.nodelist[permut1[-1]][0],myTSP.nodelist[permut1[-1]][1])
                    # 각 permutation마다 투어의 거리를 구해서 구한값이 최소이면 그때 그 투어의 리스트와 거리를 최소 투어순서, 거리로 업데이트해라
                    if totaldis < myTSP.distance:
                        myTSP.distance = totaldis
                        myTSP.tourlist = list(permut1)
                    
                    totaldis = 0  # totaldis 초기화     

                stop = timeit.default_timer()
                
                f = open(sys.argv[3],'w',newline = '') 
                data = csv.writer(f)
                data.writerow(["알고리즘","Full Enumeration"])
                data.writerow(["계산시간",(stop-start)*1000,"millisec"])
                print("계산시간",(stop-start)*1000)
                data.writerow(["전체길이",myTSP.distance])
                data.writerow(["투어순서"])
                data.writerow(["node_id","lattitude","longitude"])
                for i in myTSP.tourlist:
                    data.writerow([i,myTSP.nodelist[i][0],myTSP.nodelist[i][1]])

                f.close

                print("All work is finished")

        #------------------------------------------------------------------------------------------------------------------------------------------

            # Nearest Neighbor 알고리즘을 사용해라
            elif int(sys.argv[2]) == 2:

                start = timeit.default_timer() 

                # node 사이의 거리를 나타내는 매트릭스 dlist를 완성하라, dlist의 i행 j번째 열의 원소 a(ij)는 i노드에서 j노드로 가는 distance를 의미함
                # id가 0인 노드는 없으므로 매트릭스의 0번째행과 0번째열은 모두0이며, diagonal element도 자기자신으로 가는 거리이므로 0이다.
                for i in range(1,len(myTSP.nodelist)+1):
                    for j in range(i+1,len(myTSP.nodelist)+1):
                        dlist[i,j] = calDistance(myTSP.nodelist[i][0], myTSP.nodelist[i][1], myTSP.nodelist[j][0], myTSP.nodelist[j][1])
                        # i에서 j로 가는것과, j에서 i로 가는것은 같으므로(Symmetric tour)
                        dlist[j,i] = dlist[i,j]
                        

                # 아래의 while문을 돌릴 때의 편의를 위해 0행과 0열, diagonal element를 모두 엄청 큰 값인 100000000으로 잠시 설정함
                dlist2 = copy.deepcopy(dlist)
                dlist2[0,:] = 1000000000 # inf 대신 1000000000사용
                dlist2[:,0] = 1000000000
                for i in range(len(dlist2)):
                    dlist2[i,i] = 100000000

                count = 1 # while문을 돌릴 때 여행의 수가 (노드의 수) -1 이 될때까지 돌릴 것임
                i = 1 # 첫번째 노드부터 여행을 시작할 것임
                myTSP.tourlist.append(1) # 투어순서는 1부터 시작할것임
                

                while count < len(dlist2) - 1: # 노드의 수 - 1번 루프를 돌려라
                    totaldis += np.min(dlist2[i]) # 각 노드에서 최소거리의 노드까지 거리를 선택해라
                    i_temp = i
                    i = np.argmin(dlist2[i]) # 최소거리 노드의 id를 선택해서 다음번 i로 업데이트
                    myTSP.tourlist.append(i) # 투어리스트에 다음번 노드를 저장해라
                    count += 1
                    dlist2[:,i_temp] = 100000000 # 방문한 노드는 재방문 하면 안되므로 inf로 설정해라
                
            
                totaldis += dlist[i,1] # 마지막 노드에서 첫번째 노드까지의 거리를 더해라

                stop = timeit.default_timer()
                
                f = open(sys.argv[3],'w',newline = '')
                data = csv.writer(f)
                data.writerow(["알고리즘","Nearest Neighbor"])
                data.writerow(["계산시간",(stop-start)*1000,"millisec"])
                print("계산시간",(stop-start)*1000)
                data.writerow(["전체길이",totaldis])
                data.writerow(["투어순서"])
                data.writerow(["node_id","lattitude","longitude"])
                for i in myTSP.tourlist:
                    data.writerow([i,myTSP.nodelist[i][0],myTSP.nodelist[i][1]])

                f.close

                print("All work is finished")

        #--------------------------------------------------------------------------------------------------------------------------------------
            
            # Greedy 2 Opt 알고리즘을 사용해라
            elif int(sys.argv[2]) == 3:

                start = timeit.default_timer() 

                # node 사이의 거리를 나타내는 매트릭스 dlist를 완성하라, dlist의 i행 j번째 열의 원소 a(ij)는 i노드에서 j노드로 가는 distance를 의미함
                # id가 0인 노드는 없으므로 매트릭스의 0번째행과 0번째열은 모두0이며, diagonal element도 자기자신으로 가는 거리이므로 0이다.
                for i in range(1,len(myTSP.nodelist)+1):
                    for j in range(i+1,len(myTSP.nodelist)+1):
                        dlist[i,j] = calDistance(myTSP.nodelist[i][0], myTSP.nodelist[i][1], myTSP.nodelist[j][0], myTSP.nodelist[j][1])
                        # i에서 j로 가는것과, j에서 i로 가는것은 같으므로(Symmetric tour)
                        dlist[j,i] = dlist[i,j]
                        

                # 아래의 while문을 돌릴 때의 편의를 위해 0행과 0열, diagonal element를 모두 엄청 큰 값인 100000000으로 잠시 설정함
                dlist2 = copy.deepcopy(dlist)
                dlist2[0,:] = 1000000000 # inf 대신 1000000000사용
                dlist2[:,0] = 1000000000
                for i in range(len(dlist2)):
                    dlist2[i,i] = 100000000

                count = 1 # while문을 돌릴 때 여행의 수가 (노드의 수) -1 이 될때까지 돌릴 것임
                i = 1 # 첫번째 노드부터 여행을 시작할 것임
                myTSP.tourlist.append(1) # 투어순서는 1부터 시작할것임
                

                while count < len(dlist2) - 1: # 노드의 수 - 1번 루프를 돌려라
                    totaldis += np.min(dlist2[i]) # 각 노드에서 최소거리의 노드까지 거리를 선택해라
                    i_temp = i
                    i = np.argmin(dlist2[i]) # 최소거리 노드의 id를 선택해서 다음번 i로 업데이트
                    myTSP.tourlist.append(i) # 투어리스트에 다음번 노드를 저장해라
                    count += 1
                    dlist2[:,i_temp] = 100000000 # 방문한 노드는 재방문 하면 안되므로 inf로 설정해라
                
            
                totaldis += dlist[i,1] # 마지막 노드에서 첫번째 노드까지의 거리를 더해라
                myTSP.tourlist.append(1) # 첫번째 노드로 돌아오는것을 표시해라

        # 일단 Nearest Neighbor 알고리즘을 사용해서 feasible solution을 구해놓기
        # Node exchange가 path를 줄이는 경우, 즉시 투어 순서를 변경하고, 처음부터 다시 실행하는 알고리즘 작성   

                while True:
                    for i in range(1,len(myTSP.tourlist)-2):
                        for j in range(i+1,len(myTSP.tourlist)-1):
                            # a는 기존 path로 갔을때의 거리와 꼬인것을 푼 새로운 거리의 차(dis(i->i+1)+ dis(j->j+1) - dis(i-1->j) - dis(i->j+1)) 
                            a = dlist[myTSP.tourlist[i],myTSP.tourlist[i+1]] + dlist[myTSP.tourlist[j],myTSP.tourlist[j+1]] - dlist[myTSP.tourlist[i-1],myTSP.tourlist[j]] - dlist[myTSP.tourlist[i],myTSP.tourlist[j+1]]
                            
                # 거리에서 이득이 있으면 노드 exchange하고, 사이의 노드들은 reverse하라
                            if a > 0:
                                myTSP.tourlist[i], myTSP.tourlist[j] = myTSP.tourlist[j], myTSP.tourlist[i]
                                myTSP.tourlist[i+1:j]  = list(reversed(myTSP.tourlist[i+1:j]))
                                
                # 다 돌렸으면 while문 탈출하라
                    if i == len(myTSP.tourlist) - 3 and j == len(myTSP.tourlist) -2:
                        break
                    

                del myTSP.tourlist[-1] # 맨 마지막에 1방문하는 것은 필요없으니 지우기
                totaldis = 0 # 경로길이 초기화
                
                for i in range(len(myTSP.tourlist)-1): # 새로운 투어리스트의 total distance 구하기
                    totaldis += dlist[myTSP.tourlist[i],myTSP.tourlist[i+1]]
                
                # 맨 마지막 노드에서 처음노드로의 거리를 추가로 더해라
                totaldis += calDistance(myTSP.nodelist[1][0],myTSP.nodelist[1][1],myTSP.nodelist[myTSP.tourlist[-1]][0],myTSP.nodelist[myTSP.tourlist[-1]][1])
                stop = timeit.default_timer()
                
                f = open(sys.argv[3],'w',newline = '')
                data = csv.writer(f)
                data.writerow(["알고리즘","Nearest Neighbor with Greedy 2 Opt"])
                data.writerow(["계산시간",(stop-start)*1000,"millisec"])
                print("계산시간",(stop-start)*1000)
                data.writerow(["전체길이",totaldis])
                data.writerow(["투어순서"])
                data.writerow(["node_id","lattitude","longitude"])
                for i in myTSP.tourlist:
                    data.writerow([i,myTSP.nodelist[i][0],myTSP.nodelist[i][1]])

                f.close

                print("All work is finished")                             

        #----------------------------------------------------------------------------------------------------------------------------------------
        # Full 2 Opt 알고리즘을 사용해라

            else:

                start = timeit.default_timer() 

                # node 사이의 거리를 나타내는 매트릭스 dlist를 완성하라, dlist의 i행 j번째 열의 원소 a(ij)는 i노드에서 j노드로 가는 distance를 의미함
                # id가 0인 노드는 없으므로 매트릭스의 0번째행과 0번째열은 모두0이며, diagonal element도 자기자신으로 가는 거리이므로 0이다.
                for i in range(1,len(myTSP.nodelist)+1):
                    for j in range(i+1,len(myTSP.nodelist)+1):
                        dlist[i,j] = calDistance(myTSP.nodelist[i][0], myTSP.nodelist[i][1], myTSP.nodelist[j][0], myTSP.nodelist[j][1])
                        # i에서 j로 가는것과, j에서 i로 가는것은 같으므로(Symmetric tour)
                        dlist[j,i] = dlist[i,j]
                        

                # 아래의 while문을 돌릴 때의 편의를 위해 0행과 0열, diagonal element를 모두 엄청 큰 값인 100000000으로 잠시 설정함
                dlist2 = copy.deepcopy(dlist)
                dlist2[0,:] = 1000000000 # inf 대신 1000000000사용
                dlist2[:,0] = 1000000000
                for i in range(len(dlist2)):
                    dlist2[i,i] = 100000000

                count = 1 # while문을 돌릴 때 여행의 수가 (노드의 수) -1 이 될때까지 돌릴 것임
                i = 1 # 첫번째 노드부터 여행을 시작할 것임
                myTSP.tourlist.append(1) # 투어순서는 1부터 시작할것임
                

                while count < len(dlist2) - 1: # 노드의 수 - 1번 루프를 돌려라
                    totaldis += np.min(dlist2[i]) # 각 노드에서 최소거리의 노드까지 거리를 선택해라
                    i_temp = i
                    i = np.argmin(dlist2[i]) # 최소거리 노드의 id를 선택해서 다음번 i로 업데이트
                    myTSP.tourlist.append(i) # 투어리스트에 다음번 노드를 저장해라
                    count += 1
                    dlist2[:,i_temp] = 100000000 # 방문한 노드는 재방문 하면 안되므로 inf로 설정해라
                
            
                totaldis += dlist[i,1] # 마지막 노드에서 첫번째 노드까지의 거리를 더해라
                myTSP.tourlist.append(1) # 첫번째 노드로 돌아오는것을 표시해라

        # 일단 Nearest Neighbor 알고리즘을 사용해서 feasible solution을 구해놓기
        # Node exchange가 path를 줄이는 경우, 즉시 투어 순서를 변경하고, 처음부터 다시 실행하는 알고리즘 작성   
                

                # i번째 노드와 key번째 노드와의 exchange로 value값만큼의 Gain이 생김
                hubolist = {}

                while True:
                    for i in range(1,len(myTSP.tourlist)-2):
                        for j in range(i+1,len(myTSP.tourlist)-1):
                            # a는 기존 path로 갔을때의 거리(dis(i->i+1)+ dis(j->j+1) - dis(i-1->j) - dis(i->j+1)) - 새로운 path로 갔을때의 거fl
                            a = dlist[myTSP.tourlist[i],myTSP.tourlist[i+1]] + dlist[myTSP.tourlist[j],myTSP.tourlist[j+1]] - dlist[myTSP.tourlist[i-1],myTSP.tourlist[j]] - dlist[myTSP.tourlist[i],myTSP.tourlist[j+1]]
                            
                # 거리에서 이득이 있으면 그 노드들을 hubolist에 넣어라
                            if a > 0:
                                hubolist[myTSP.tourlist[j]] = a

                        if len(hubolist) != 0:
                            b = list(hubolist.values()).index(max(list(hubolist.values())))
                            c = list(hubolist.keys())[b] # 가장 이득이 큰 node의 id
                            d = myTSP.tourlist.index(c)

                            # 가장 이득이 큰 노드에 대해서  노드 exchange하고, 그 사이의 노드들은 reverse하라
                            myTSP.tourlist[i], myTSP.tourlist[d] = myTSP.tourlist[d], myTSP.tourlist[i]
                            myTSP.tourlist[i+1:d]  = list(reversed(myTSP.tourlist[i+1:d]))
                        

                        hubolist.clear()

                    if i == len(myTSP.tourlist) - 3 and j == len(myTSP.tourlist) -2:
                        break
        

                del myTSP.tourlist[-1] # 맨 마지막에 1방문하는 것은 필요없으니 지우기
                totaldis = 0 # 경로길이 초기화
                
                for i in range(len(myTSP.tourlist)-1): # 새로운 투어리스트의 total distance 구하기
                    totaldis += dlist[myTSP.tourlist[i],myTSP.tourlist[i+1]]
                # 맨 마지막 노드에서 처음 노드로의 거리를 더해라 
                totaldis += calDistance(myTSP.nodelist[1][0],myTSP.nodelist[1][1],myTSP.nodelist[myTSP.tourlist[-1]][0],myTSP.nodelist[myTSP.tourlist[-1]][1])
                stop = timeit.default_timer()
                
                f = open(sys.argv[3],'w',newline = '')
                data = csv.writer(f)
                data.writerow(["알고리즘","Nearest Neighbor with Full 2 Opt"])
                data.writerow(["계산시간",(stop-start)*1000,"millisec"])
                print("계산시간",(stop-start)*1000)
                data.writerow(["전체길이",totaldis])
                data.writerow(["투어순서"])
                data.writerow(["node_id","lattitude","longitude"])
                for i in myTSP.tourlist:
                    data.writerow([i,myTSP.nodelist[i][0],myTSP.nodelist[i][1]])

                f.close

                print("All work is finished")         

    
        else:
            raise Exception()

    except Exception as e:
        print("제대로 입력을 안하셨습니다",e)


          
        



    