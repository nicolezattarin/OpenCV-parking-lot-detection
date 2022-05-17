#include "parking.h"

Parking :: Parking(){
    m_isEmpty = true;
    int m_id = -1;
    float m_x = -1;
    float m_y = -1;
    float m_width = -1;
    float m_height = -1; 
}

Parking :: Parking(int id, int x, int y, int width, int height){
    m_isEmpty = true;
    m_id = id;
    m_x = x;
    m_y = y;
    m_width = width;
    m_height = height;
}

// setters/getters
void Parking :: setEmpty(bool isEmpty){
    m_isEmpty = isEmpty;
}

bool Parking :: isEmpty(){
    return m_isEmpty;
}

void Parking :: setId(int id){
    m_id = id;
}

int Parking :: getId(){
    return m_id;
}

void Parking :: setX(int x){
    m_x = x;
}

int Parking :: getX(){
    return m_x;
}

void Parking :: setY(int y){
    m_y = y;
}

int Parking :: getY(){
    return m_y;
}

void Parking :: setWidth(int width){
    m_width = width;
}

int Parking :: getWidth(){
    return m_width;
}

void Parking :: setHeight(int height){
    m_height = height;
}

int Parking :: getHeight(){
    return m_height;
}