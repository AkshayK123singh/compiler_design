int multi(int a, int b){
    int c;
    c=a*b;
    return c;
}
int main(){
    int a;
    scanf("%d",&a);
    printf("\n");
    int b;
    scanf("%d",&b);
    int c;
    c=multi(a,b);
    printf("%d\n",c);
    return 0;
}