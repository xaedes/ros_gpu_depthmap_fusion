#pragma once
#include <string>

// https://stackoverflow.com/a/2548212
std::string& replace_string_inplace(std::string& str, const std::string& remove, const std::string& insert) 
{
    std::string::size_type pos = 0;
    while ((pos = str.find(remove, pos)) != std::string::npos)
    {
        str.replace(pos, remove.size(), insert);
        pos++;
    }

    return str;
}

std::string replace_strings_inplace(std::string& str, const std::vector<std::string>& remove, const std::vector<std::string>& insert)
{
    assert(remove.size() == insert.size());
    for (size_t i = 0; i<remove.size(); i++)
    {
        replace_string_inplace(str, remove[i], insert[i]);
    }
}
