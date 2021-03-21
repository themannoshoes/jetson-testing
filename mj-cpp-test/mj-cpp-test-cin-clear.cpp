#include <iostream>
#include <fstream>

#define ITEM 10
using namespace std;
int main(void)
{
    int num[ITEM];
    std::cout << "please input"<<  ITEM <<" int \n \n ";
    for(int i =0;i < ITEM;i++)
    {
        std::cout << "please input No." << i + 1 << "number value : \n";
        while( !(std::cin >> num[i])){
            std::cin.clear();
            std::cin.ignore(100, '\n');
            std::cout << "wrong value you have input, input again please!";
        }
    }
    int total = 0;
    for( int j =0;j < ITEM;j++){
        total += num[j];

    }
    std::cout << "total value is " << total << endl;
    return 0;

}


