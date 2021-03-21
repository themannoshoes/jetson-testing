#include <iostream>
#include <fstream>

using namespace std;
int main(void)
{
    ofstream out("test.txt");
    if(!out)
    {

        cerr << "failed to open" << endl;
        return 0;
    }
    for(int i =10; i > 0;i--){
            out << i;
    }
    out << endl;
    out.close();
    return 0;

}